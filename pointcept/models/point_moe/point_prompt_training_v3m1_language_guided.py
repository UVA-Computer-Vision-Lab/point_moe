"""
Point Prompt Training

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
        Xuweiyi Chen (xuweic@email.virginia.edu)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria


@MODELS.register_module("PPTMoE-v0m0")
class PointPromptMoETraining(nn.Module):
    """
    PointPromptTraining provides Data-driven Context and enables multi-dataset training with
    Language-driven Categorical Alignment.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        backbone_out_channels=96,
        context_channels=256,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
        template="[x]",
        clip_model="ViT-B/16",
        # fmt: off
        class_name=(
            "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
            "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
            "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
            "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
            "clutter", "otherstructure", "otherfurniture", "otherprop",
        ),
        valid_index=(
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
            (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
        ),
        # fmt: on
        backbone_mode=False,
    ):
        super().__init__()
        assert len(conditions) == len(valid_index)
        assert backbone.type in ["PTMoE-v0m0", "SpUNet-v1m3", "PT-v2m3", "PT-v3m1", "PTMoE-v4m1", "PTMoE-v1m1"]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.conditions = conditions
        self.valid_index = valid_index
        # self.embedding_table = nn.Embedding(len(conditions), context_channels)
        self.backbone_mode = backbone_mode
        
        if not self.backbone_mode:
            import clip

            clip_model, _ = clip.load(
                clip_model, device="cpu", download_root="./.cache/clip"
            )
            clip_model.requires_grad_(False)
            class_prompt = [template.replace("[x]", name) for name in class_name]
            class_token = clip.tokenize(class_prompt)
            class_embedding = clip_model.encode_text(class_token)
            class_embedding = class_embedding / class_embedding.norm(
                dim=-1, keepdim=True
            )
            self.register_buffer("class_embedding", class_embedding)
            self.proj_head = nn.Linear(
                backbone_out_channels, clip_model.text_projection.shape[1]
            )
            self.logit_scale = clip_model.logit_scale
        
    def forward(self, data_dict):
        condition_list = data_dict["condition"]
        offsets = data_dict["offset"]
        device = offsets.device
        N = offsets[-1].item()

        dataset_name_to_id = {name: idx for idx, name in enumerate(self.conditions)}
        dataset_ids = torch.zeros((N, 1), dtype=torch.long, device=device)

        all_indices = torch.arange(N, device=device).unsqueeze(1)
        dataset_to_indices = {}

        for seg_idx, dataset in enumerate(condition_list):
            start_idx = offsets[seg_idx - 1].item() if seg_idx > 0 else 0
            end_idx = offsets[seg_idx].item()

            indices = all_indices[start_idx:end_idx]
            dataset_id = dataset_name_to_id[dataset]
            dataset_ids[start_idx:end_idx] = dataset_id

            if dataset not in dataset_to_indices:
                dataset_to_indices[dataset] = []

            dataset_to_indices[dataset].append(indices.squeeze(1))

        data_dict["dataset_ids"] = dataset_ids
        
        # === Step 2: Backbone forward ===
        point = self.backbone(data_dict)
        feat = point.feat if isinstance(point, Point) else point

        if self.backbone_mode:
            return feat
        
        # === Step 3: Normalize features ===
        feat = self.proj_head(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        seg_logits_list = []
        losses = []
        has_segment = "segment" in data_dict

        for dataset_name, index_list in dataset_to_indices.items():
            valid_classes = self.valid_index[self.conditions.index(dataset_name)]
            valid_class_indices = torch.tensor(valid_classes, device=feat.device)

            dataset_indices = torch.cat(index_list, dim=0)
            feat_i = feat[dataset_indices]

            class_embedding_i = self.class_embedding[valid_class_indices]
            sim_i = feat_i @ class_embedding_i.T

            logit_scale = self.logit_scale.exp()
            seg_logits_i = logit_scale * sim_i
            seg_logits_list.append(seg_logits_i)

            if self.training or has_segment:
                segment_labels = data_dict["segment"][dataset_indices]
                loss = self.criteria(seg_logits_i, segment_labels)
                losses.append(loss)

        if self.training:
            total_loss = (
                torch.stack(losses).mean()
                if losses
                else torch.tensor(0.0, device=feat.device)
            )
            return dict(loss=total_loss)

        elif has_segment:
            total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=feat.device)
            return dict(loss=total_loss, seg_logits=seg_logits_list)

        seg_logits = torch.cat(seg_logits_list, dim=0)  # Shape: [N, num_classes]
        return dict(seg_logits=seg_logits)

# def forward(self, data_dict):
#     condition_list = data_dict["condition"]
#     offsets = data_dict["offset"]
#     segment_labels = data_dict["segment"]  # Shape: [N]

#     # Step 1: Create dataset_id mapping before passing through the backbone
#     dataset_name_to_id = {name: idx for idx, name in enumerate(self.conditions)}
#     N = offsets[-1].item()  # Total number of points
#     dataset_ids = torch.zeros(N, dtype=torch.long, device=segment_labels.device)

#     all_indices = torch.arange(N, device=segment_labels.device)
    
#     # Create dataset ID mapping in one operation
#     dataset_slices = torch.cat([torch.full((offsets[i] - (offsets[i-1] if i > 0 else 0),), 
#                                            dataset_name_to_id[dataset], 
#                                            device=segment_labels.device, dtype=torch.long) 
#                                 for i, dataset in enumerate(condition_list)])

#     dataset_ids[:] = dataset_slices
#     data_dict["dataset_ids"] = dataset_ids

#     # Step 2: Pass data through the backbone
#     point = self.backbone(data_dict)
#     has_aux_loss = False
#     if isinstance(point, tuple):
#         has_aux_loss = True
#         point, aux_loss = point

#     feat = point.feat if isinstance(point, Point) else point

#     if self.backbone_mode:
#         return feat

#     # Normalize feature embeddings
#     feat = self.proj_head(feat)
#     feat = feat / feat.norm(dim=-1, keepdim=True)

#     # Step 3: Compute logits and loss efficiently using `torch.gather`
#     valid_class_indices = torch.cat(
#         [torch.tensor(self.valid_index[self.conditions.index(dataset)], device=feat.device) 
#          for dataset in condition_list])

#     # Gather valid class embeddings in a single operation
#     class_embedding_i = self.class_embedding[valid_class_indices]

#     # Compute similarity using `torch.gather`
#     sim_i = torch.bmm(feat.unsqueeze(1), class_embedding_i.unsqueeze(2)).squeeze(1)  

#     logit_scale = self.logit_scale.exp()
#     seg_logits_i = logit_scale * sim_i

#     # Compute loss using batched segment labels
#     total_loss = self.criteria(seg_logits_i, segment_labels)

#     if has_aux_loss:
#         total_loss += aux_loss

#     if self.training:
#         return dict(loss=total_loss)

#     elif "segment" in data_dict:
#         return dict(loss=total_loss, seg_logits=seg_logits_i)

#     return dict(seg_logits=seg_logits_i)