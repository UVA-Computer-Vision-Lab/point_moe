"""
Point MoE - V1 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
        Xuweiyi Chen (xuweic@email.virginia.edu)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

# from .layer import MoE
from .moe_layer import MoELayer


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        # MoE parameters
        use_moe_proj=False,
        use_moe_attn=False,  # New argument to control MoE in attention
        num_experts=4,
        topk=1,
        moe_use_residual=False,
        aux_loss_alpha=0.01,
        n_intermediate_size=1,
        act_fn="relu",
        act_fn_params=None,
        proj_ratio=1.0,  # Added parameter to control projection size
        domain_guided=False,
        context_channels=16,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        if use_moe_proj:
            # Create MoELayer config for projection
            config = type(
                "Config",
                (),
                {
                    "hidden_size": channels,
                    "intermediate_size": int(channels * n_intermediate_size),
                    "moe_intermediate_size": int(channels * n_intermediate_size),
                    "num_experts_per_tok": topk,
                    "n_routed_experts": num_experts,
                    "n_shared_experts": 1 if moe_use_residual else 0,
                    "aux_loss_alpha": aux_loss_alpha,
                    "norm_topk_prob": True,
                    "act_fn": act_fn,
                    "act_fn_params": act_fn_params,
                    "domain_guided": domain_guided,
                    "context_channels": context_channels,
                },
            )
            self.proj = MoELayer(config)
            self.is_moe_proj = True
        else:
            # Use expanded projection if proj_ratio > 1
            if proj_ratio > 1.0:
                # Create larger intermediate dimensions with expansion and contraction
                self.proj = nn.Sequential(
                    nn.Linear(channels, int(channels * proj_ratio)),
                    nn.GELU(),
                    nn.Linear(int(channels * proj_ratio), channels),
                )
            else:
                # Standard projection
                self.proj = torch.nn.Linear(channels, channels)
            self.is_moe_proj = False
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

        if use_moe_attn:
            # Create MoELayer config for attention
            attn_config = type(
                "Config",
                (),
                {
                    "hidden_size": channels,
                    "intermediate_size": int(channels * n_intermediate_size),
                    "moe_intermediate_size": int(channels * n_intermediate_size),
                    "num_experts_per_tok": topk,
                    "n_routed_experts": num_experts,
                    "n_shared_experts": 1 if moe_use_residual else 0,
                    "aux_loss_alpha": aux_loss_alpha,
                    "norm_topk_prob": True,
                    "act_fn": act_fn,
                    "act_fn_params": act_fn_params,
                    "domain_guided": domain_guided,
                    "context_channels": context_channels,
                },
            )
            self.q_moe = MoELayer(attn_config)
            self.kv_proj = torch.nn.Linear(channels, channels * 2, bias=qkv_bias)
            self.is_moe_attn = True
        else:
            self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
            self.is_moe_attn = False

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        if self.is_moe_attn:
            # Use MoE directly for query projection
            q_proj = self.q_moe(point.feat)
            kv = self.kv_proj(point.feat)
            q_proj = q_proj[order]
            k, v = kv[order].chunk(2, dim=-1)
            q = q_proj

            if not self.enable_flash:
                # encode and reshape q and kv
                q = q.reshape(-1, K, H, C // H)  # (N', K, H, C')
                k, v = (
                    kv.reshape(-1, K, 2, H, C // H).permute(2, 0, 1, 3, 4).unbind(dim=0)
                )  # (N', K, H, C')

                # attn
                if self.upcast_attention:
                    q = q.float()
                    k = k.float()
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
                if self.enable_rpe:
                    attn = attn + self.rpe(self.get_rel_pos(point, order))
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(q.dtype)
                feat = (attn @ v).transpose(1, 2).reshape(-1, C)
            else:
                qkv_stacked = torch.cat(
                    [q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)], dim=1
                ).reshape(-1, 3, H, C // H)
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv_stacked.half(),
                    cu_seqlens,
                    max_seqlen=self.patch_size,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)
                feat = feat.to(q.dtype)  # Ensure correct dtype
        else:
            # Original QKV projection for non-MoE case
            qkv = self.qkv(point.feat)[order]

            if not self.enable_flash:
                # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
                q, k, v = (
                    qkv.reshape(-1, K, 3, H, C // H)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(dim=0)
                )
                # attn
                if self.upcast_attention:
                    q = q.float()
                    k = k.float()
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
                if self.enable_rpe:
                    attn = attn + self.rpe(self.get_rel_pos(point, order))
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(qkv.dtype)
                feat = (attn @ v).transpose(1, 2).reshape(-1, C)
            else:
                feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                    qkv.half().reshape(-1, 3, H, C // H),
                    cu_seqlens,
                    max_seqlen=self.patch_size,
                    dropout_p=self.attn_drop if self.training else 0,
                    softmax_scale=self.scale,
                ).reshape(-1, C)
                feat = feat.to(qkv.dtype)

        feat = feat[inverse]

        # ffn
        if self.is_moe_proj:
            point.feat = feat
            feat = self.proj(point)
        else:
            feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        # MoE parameters
        use_moe_mlp=False,
        use_moe_proj=False,
        num_experts=4,
        topk=1,
        moe_use_residual=False,
        aux_loss_alpha=0,
        n_intermediate_size=1,
        act_fn="relu",
        act_fn_params=None,
        use_moe_attn=False,
        proj_ratio=1.0,  # Added parameter for projection ratio
        domain_guided=False,
        context_channels=16,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            # Pass MoE parameters for projection
            use_moe_proj=use_moe_proj,
            num_experts=num_experts,
            topk=topk,
            moe_use_residual=moe_use_residual,
            aux_loss_alpha=aux_loss_alpha,
            n_intermediate_size=n_intermediate_size,
            act_fn=act_fn,
            act_fn_params=act_fn_params,
            use_moe_attn=use_moe_attn,
            proj_ratio=proj_ratio,  # Pass the projection ratio
            domain_guided=domain_guided,
            context_channels=context_channels,
        )
        self.norm2 = PointSequential(norm_layer(channels))

        if use_moe_mlp:
            config = type(
                "Config",
                (),
                {
                    "hidden_size": channels,
                    "intermediate_size": int(channels * n_intermediate_size),
                    "moe_intermediate_size": int(channels * n_intermediate_size),
                    "num_experts_per_tok": topk,
                    "n_routed_experts": num_experts,
                    "n_shared_experts": 1 if moe_use_residual else 0,
                    "aux_loss_alpha": aux_loss_alpha,
                    "norm_topk_prob": True,
                    "act_fn": act_fn,
                    "act_fn_params": act_fn_params,
                    "domain_guided": domain_guided,
                    "context_channels": context_channels,
                },
            )
            self.mlp = PointSequential(MoELayer(config))
        else:
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        _, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "contexts" in point.keys():
            point_dict["contexts"] = point.contexts

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PTMoE-v0m0")
class PointTransformerMoE(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        proj_ratio=1.0,  # Added parameter for projection layer ratio
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        # MoE parameters
        use_moe=False,
        use_moe_mlp=True,  # Whether to use MoE in Block MLP
        use_moe_proj=False,  # Whether to use MoE in attention projection
        num_experts=4,
        topk=1,
        moe_use_residual=False,
        moe_layers=None,  # None means all layers, or provide list of indices
        use_ln=False,
        use_RMS=False,  # Whether to use RMSNorm instead of LayerNorm
        rms_norm_partial=-1.0,  # Partial RMSNorm parameter, value [0, 1]
        rms_norm_eps=1e-8,  # Epsilon for RMSNorm
        rms_norm_bias=False,  # Whether to use bias in RMSNorm
        enable_enc=True,
        enable_dec=True,
        aux_loss_alpha=0,
        n_intermediate_size=1,
        # Activation function parameters
        act_fn="relu",  # Options: "relu", "gelu", "silu", "leaky_relu", "elu", "mish"
        act_fn_params=None,  # Additional parameters for activation function
        use_moe_attn=False,
        domain_guided=False,
        context_channels=16,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.use_moe = use_moe
        self.use_moe_mlp = use_moe_mlp
        self.use_moe_proj = use_moe_proj
        self.num_experts = num_experts
        self.topk = topk
        self.moe_use_residual = moe_use_residual
        self.moe_layers = moe_layers
        self.use_ln = use_ln
        self.use_RMS = use_RMS
        self.rms_norm_partial = rms_norm_partial
        self.rms_norm_eps = rms_norm_eps
        self.rms_norm_bias = rms_norm_bias
        self.proj_ratio = proj_ratio  # Store the projection ratio
        self.domain_guided = domain_guided
        self.context_channels = context_channels

        # Ensure only one normalization method is selected
        assert not (
            self.use_ln and self.use_RMS
        ), "Cannot use both LayerNorm and RMSNorm at the same time"

        self.enable_enc = enable_enc
        self.enable_dec = enable_dec
        self.aux_loss_alpha = aux_loss_alpha
        self.n_intermediate_size = n_intermediate_size
        self.act_fn = act_fn
        self.act_fn_params = act_fn_params
        self.use_moe_attn = use_moe_attn
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if self.use_RMS:
            def ln_layer(d):
                return RMSNorm(
                    d=d,
                    p=self.rms_norm_partial,
                    eps=self.rms_norm_eps,
                    bias=self.rms_norm_bias,
                )
        else:
            ln_layer = nn.LayerNorm

        # activation layers
        act_layer = nn.GELU
        if self.use_ln or self.use_RMS:
            self.embedding = Embedding(
                in_channels=in_channels,
                embed_channels=enc_channels[0],
                norm_layer=ln_layer,
                act_layer=act_layer,
            )
        else:
            self.embedding = Embedding(
                in_channels=in_channels,
                embed_channels=enc_channels[0],
                norm_layer=bn_layer,
                act_layer=act_layer,
            )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        layer_idx = 0
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                if self.use_ln or self.use_RMS:
                    enc.add(
                        SerializedPooling(
                            in_channels=enc_channels[s - 1],
                            out_channels=enc_channels[s],
                            stride=stride[s - 1],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                        ),
                        name="down",
                    )
                else:
                    enc.add(
                        SerializedPooling(
                            in_channels=enc_channels[s - 1],
                            out_channels=enc_channels[s],
                            stride=stride[s - 1],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                        ),
                        name="down",
                    )
            for i in range(enc_depths[s]):
                # Determine if this layer should use MoE
                apply_moe = self.use_moe and (
                    self.moe_layers is None or layer_idx in self.moe_layers
                )
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        # MoE parameters
                        use_moe_mlp=apply_moe and self.use_moe_mlp and self.enable_enc,
                        use_moe_proj=apply_moe
                        and self.use_moe_proj
                        and self.enable_enc,
                        num_experts=self.num_experts,
                        topk=self.topk,
                        moe_use_residual=self.moe_use_residual,
                        aux_loss_alpha=self.aux_loss_alpha,
                        n_intermediate_size=self.n_intermediate_size,
                        act_fn=self.act_fn,
                        act_fn_params=self.act_fn_params,
                        use_moe_attn=self.use_moe_attn,
                        proj_ratio=self.proj_ratio,
                        domain_guided=self.domain_guided,
                        context_channels=self.context_channels,
                    ),
                    name=f"block{i}",
                )
                layer_idx += 1
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            layer_idx = 0
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                if self.use_ln or self.use_RMS:
                    dec.add(
                        SerializedUnpooling(
                            in_channels=dec_channels[s + 1],
                            skip_channels=enc_channels[s],
                            out_channels=dec_channels[s],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                        ),
                        name="up",
                    )
                else:
                    dec.add(
                        SerializedUnpooling(
                            in_channels=dec_channels[s + 1],
                            skip_channels=enc_channels[s],
                            out_channels=dec_channels[s],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                        ),
                        name="up",
                    )
                for i in range(dec_depths[s]):
                    # Determine if this layer should use MoE
                    apply_moe = self.use_moe and (
                        self.moe_layers is None or layer_idx in self.moe_layers
                    )
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            # MoE parameters
                            use_moe_mlp=apply_moe
                            and self.use_moe_mlp
                            and self.enable_dec,
                            use_moe_proj=apply_moe
                            and self.use_moe_proj
                            and self.enable_dec,
                            num_experts=self.num_experts,
                            topk=self.topk,
                            moe_use_residual=self.moe_use_residual,
                            aux_loss_alpha=self.aux_loss_alpha,
                            n_intermediate_size=self.n_intermediate_size,
                            act_fn=self.act_fn,
                            act_fn_params=self.act_fn_params,
                            use_moe_attn=self.use_moe_attn,
                            proj_ratio=self.proj_ratio,
                            domain_guided=self.domain_guided,
                            context_channels=self.context_channels,
                        ),
                        name=f"block{i}",
                    )
                    layer_idx += 1
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if not self.cls_mode:
            point = self.dec(point)

        return point
