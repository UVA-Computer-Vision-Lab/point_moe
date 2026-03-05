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
from .point_detector import (
    PointCloudDetector,
)

@MODELS.register_module("PPTMoE-v0m1")
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
        context_channels=16,
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
        enable_domain_guided=False,
        enable_generic_class=False,
        detector_weight=None,
        inference_mode="gt",
    ):
        super().__init__()
        assert len(conditions) == len(valid_index)
        assert backbone.type in [
            "PTMoE-v0m0",
            "SpUNet-v1m3",
            "PT-v2m3",
            "PT-v3m1",
            "PTMoE-v4m1",
            "PTMoE-v1m1",
        ]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.conditions = conditions
        self.valid_index = valid_index
        self.backbone_mode = backbone_mode
        self.enable_generic_class = enable_generic_class
        self.enable_domain_guided = enable_domain_guided
        self.detector_weight = detector_weight
        self.inference_mode = inference_mode
        assert self.inference_mode in ["gt", "one_for_all", "generic"]

        if self.enable_domain_guided:
            if self.enable_generic_class:
                self.conditions = conditions + ("generic",)
                self.embedding_table = nn.Embedding(len(conditions) + 1, context_channels)
            else:
                self.embedding_table = nn.Embedding(len(conditions), context_channels)

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

        if self.inference_mode == "one_for_all":
            excluded = {"Matterport3D", "Waymo"}
            detector_conditions = tuple(c for c in conditions if c not in excluded)
            self.detector = PointCloudDetector(
                in_channels=6,
                num_classes=len(detector_conditions),
                conditions=detector_conditions,
            )
            state = torch.load(detector_weight, map_location="cpu")

            if "state_dict" in state:
                state = state["state_dict"]

            self.detector.load_state_dict(
                state, strict=False
            )

            self.detector.eval()
            self.detector.to("cuda")

    def forward(self, data_dict):
        condition_list = data_dict["condition"]
        reserved_condition_list = condition_list
        if self.enable_domain_guided:
            # Get a list of contexts from the embedding table based on conditions
            context_indices = [self.conditions.index(condition) for condition in condition_list]
            context_indices = torch.tensor(context_indices, device=data_dict["coord"].device)
            
            if self.training:
                if self.enable_generic_class:
                    generic_idx = len(self.conditions) - 1
                    num_indices = len(context_indices)
                    num_generic = max(1, num_indices // 6)  # At least 1 index
                    if torch.rand(1).item() < 0.5:
                        generic_positions = torch.randperm(num_indices)[:num_generic]
                        context_indices[generic_positions] = generic_idx
            else:
                if self.inference_mode == "gt":
                    assert "Matterport3D" not in condition_list and "Waymo" not in condition_list
                    # data_dict["condition"] = condition_list
                    context_indices = [self.conditions.index(condition) for condition in condition_list]
                    context_indices = torch.tensor(context_indices, device=data_dict["coord"].device)
                elif self.inference_mode == "one_for_all":
                    condition_dict = self.detector(data_dict)
                    conditions = condition_dict["pred_condition"]
                    print(conditions)
                    context_indices = [self.conditions.index(condition) for condition in conditions]
                    context_indices = torch.tensor(context_indices, device=data_dict["coord"].device)
                elif self.inference_mode == "generic":
                    context_indices = torch.tensor([len(self.conditions) - 1], device=data_dict["coord"].device)
            
            
            contexts = self.embedding_table(context_indices)
        
            # Store contexts in data_dict for use in backbone
            data_dict["contexts"] = contexts
        
        offsets = data_dict["offset"]
        segment_labels = data_dict["segment"]

        # Step 1: Create dataset_id mapping before passing through the backbone
        dataset_name_to_id = {name: idx for idx, name in enumerate(self.conditions)}
        N = offsets[-1].item()  # Total number of points
        dataset_ids = torch.zeros(
            (N, 1), dtype=torch.long, device=segment_labels.device
        )

        all_indices = torch.arange(N, device=segment_labels.device).unsqueeze(1)
        dataset_to_indices = {}

        for seg_idx, dataset in enumerate(reserved_condition_list):
            start_idx = offsets[seg_idx - 1].item() if seg_idx > 0 else 0
            end_idx = offsets[seg_idx].item()  # This is exclusive

            indices = all_indices[start_idx:end_idx]  # Extract indices for this dataset
            dataset_id = dataset_name_to_id[dataset]
            dataset_ids[start_idx:end_idx] = dataset_id  # Assign dataset index

            if dataset not in dataset_to_indices:
                dataset_to_indices[dataset] = []

            dataset_to_indices[dataset].append(indices.squeeze(1))

        # Assign dataset_ids to data_dict before passing through the backbone
        data_dict["dataset_ids"] = dataset_ids

        # Step 2: Pass data through the backbone
        point = self.backbone(data_dict)
        feat = point.feat if isinstance(point, Point) else point

        if self.backbone_mode:
            return feat

        # Normalize feature embeddings
        feat = self.proj_head(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        seg_logits_list = []
        losses = []

        # Step 3: Compute logits and loss per dataset
        for dataset_name, index_list in dataset_to_indices.items():
            valid_classes = self.valid_index[self.conditions.index(dataset_name)]
            valid_class_indices = torch.tensor(valid_classes, device=feat.device)

            # Concatenate all indices for this dataset
            dataset_indices = torch.cat(index_list, dim=0)
            feat_i = feat[dataset_indices]
            segment_i = segment_labels[dataset_indices]

            # Compute similarity using only valid class embeddings
            class_embedding_i = self.class_embedding[valid_class_indices]
            sim_i = feat_i @ class_embedding_i.T

            logit_scale = self.logit_scale.exp()
            seg_logits_i = logit_scale * sim_i
            seg_logits_list.append(seg_logits_i)

            loss = self.criteria(seg_logits_i, segment_i)
            losses.append(loss)

        total_loss = (
            torch.stack(losses).mean()
            if losses
            else torch.tensor(0.0, device=feat.device)
        )

        if self.training:
            return dict(loss=total_loss)

        elif "segment" in data_dict:
            return dict(loss=total_loss, seg_logits=seg_logits_list)

        # Test mode: return logits
        return dict(seg_logits=seg_logits_list)
