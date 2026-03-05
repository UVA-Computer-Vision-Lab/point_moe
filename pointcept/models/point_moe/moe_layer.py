import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pointcept.models.utils.structure import Point

class MoEMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = self._get_activation_fn(config.act_fn, config.act_fn_params)

    def _get_activation_fn(self, act_fn="relu", act_fn_params=None):
        """Get activation function based on configuration"""
        act_fn = act_fn.lower()
        act_fn_params = act_fn_params or {}
        
        if act_fn == "relu":
            return F.relu
        elif act_fn == "gelu":
            return F.gelu
        elif act_fn == "silu":
            return F.silu
        elif act_fn == "leaky_relu":
            negative_slope = act_fn_params.get("negative_slope", 0.01)
            return lambda x: F.leaky_relu(x, negative_slope=negative_slope)
        elif act_fn == "elu":
            alpha = act_fn_params.get("alpha", 1.0)
            return lambda x: F.elu(x, alpha=alpha)
        elif act_fn == "mish":
            def mish(x):
                return x * torch.tanh(F.softplus(x))
            return mish
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")


    def forward(self, x):
        # breakpoint()
        # Store original dtype to ensure we return the same dtype
        original_dtype = x.dtype

        # Perform computation
        result = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        # Convert back to original dtype if needed
        if result.dtype != original_dtype:
            result = result.to(original_dtype)

        return result


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.top_k = config.num_experts_per_tok
        self.n_experts = config.n_routed_experts
        self.alpha = config.aux_loss_alpha
        self.norm_topk_prob = config.norm_topk_prob
        self.domain_guided = config.domain_guided

        if self.domain_guided:
            self.weight = nn.Parameter(torch.empty((self.n_experts, config.hidden_size + config.context_channels)))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = nn.Parameter(torch.empty((self.n_experts, config.hidden_size)))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Just one buffer to store the accumulated scores
        self.register_buffer("score_accumulator", torch.zeros(self.n_experts, dtype=torch.float))

    def forward(self, hidden_states):
        if isinstance(hidden_states, Point):
            hidden_states = hidden_states.feat
        logits = F.linear(hidden_states, self.weight)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.norm_topk_prob and self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        aux_loss = None
        if self.training and self.alpha > 0:
            mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.n_experts).float()
            ce = mask_ce.mean(0)
            aux_loss = (scores.mean(0) * ce * self.n_experts).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        ctx.required_aux_loss = loss.requires_grad
        ctx.dtype = loss.dtype
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = (
            torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
            if ctx.required_aux_loss
            else None
        )
        return grad_output, grad_loss


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                MoEMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.domain_guided = config.domain_guided
        self.gate = MoEGate(config)

        if config.n_shared_experts:
            shared_intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = MoEMLP(
                config, intermediate_size=shared_intermediate_size
            )
        else:
            self.shared_experts = None
        
        self.register_buffer("expert_token_counter", torch.zeros(config.n_routed_experts, dtype=torch.long))

    def forward(self, point):
        if isinstance(point, Point):
            hidden_states = point.feat
        else:
            hidden_states = point
        if self.domain_guided:
            contexts = point.contexts
            offset = point.offset
            
            # Prepend 0 to offset if missing
            if offset[0] != 0:
                offset = torch.cat((torch.tensor([0], device=offset.device), offset))
            
            # Calculate the number of tokens in each segment
            segment_lengths = offset[1:] - offset[:-1]
            # Repeat contexts to match the number of tokens in each segment
            repeated_contexts = contexts.repeat_interleave(segment_lengths, dim=0)
            assert repeated_contexts.size(0) == hidden_states.size(0), \
                "Mismatch in token and context alignment"
            # Concatenate hidden states and repeated contexts
            gate_hidden_states = torch.cat([hidden_states, repeated_contexts], dim=1)
        else:
            gate_hidden_states = hidden_states
                
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(gate_hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(hidden_states[mask])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(orig_shape)
            if self.config.aux_loss_alpha > 0:
                y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(orig_shape)

        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_out = self.experts[i](x[exp_token_idx])
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.unsqueeze(-1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache
