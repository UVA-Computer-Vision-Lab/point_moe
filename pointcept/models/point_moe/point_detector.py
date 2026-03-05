"""
Point Cloud Dataset Detector

Author: Xuweiyi Chen (xuweic@email.virginia.edu)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.utils.rnn as rnn_utils
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointSequential
from pointcept.models.builder import MODELS

class DatasetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.net = PointSequential(  # Feature extraction network
            spconv.SubMConv3d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                bias=True,
                indice_key="classifier_3x3",
            ),
            spconv.SubMConv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                bias=True,
                indice_key="classifier_1x1",
            ),
        )
        # Modified classifier to handle concatenated features (256 * 2 = 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # First reduce the concatenated features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.1)  # Add label smoothing

    def forward(self, data_dict, offsets, dataset_labels=None):
        """
        Args:
            data_dict: dictionary containing point data. Expected keys include "feat", "coord", and "batch".
            offsets: 1D tensor containing the end indices for each dataset segment.
            dataset_labels: Optional tensor of shape (num_datasets,) with ground-truth dataset indices.
        Returns:
            padded_feats: Padded tensor of processed features for each dataset segment, shape (num_datasets, max_N, 256)
            dataset_logits: Tensor of shape (num_datasets, num_classes) from the classifier.
            loss: Classification loss if dataset_labels is provided.
        """
        # breakpoint()
        point_features_list = []
        start_idx = 0

        for end_idx in offsets:
            truncated_data = {}
            keys_to_truncate = ["coord", "grid_coord", "offset", "grid_coord", "feat"]
            for key in keys_to_truncate:
                if key in data_dict:
                    truncated_data[key] = data_dict[key][start_idx:end_idx]

            truncated_data["offset"] = torch.tensor(
                [truncated_data["feat"].shape[0]], device=truncated_data["feat"].device
            )
            truncated_point = Point(truncated_data)
            truncated_point.sparsify()
            processed_segment = self.net(truncated_point).feat  # Expect shape: (N_i, 256)
            point_features_list.append(processed_segment)

            start_idx = end_idx.item()

        padded_feats = rnn_utils.pad_sequence(
            point_features_list, batch_first=True
        )  # (num_datasets, max_N, 256)

        # Mean and max pooling
        mean_pool = padded_feats.mean(dim=1)  # (num_datasets, 256)
        max_pool = padded_feats.max(dim=1)[0]  # (num_datasets, 256)
        # Concatenate mean and max pooling
        segment_representation = torch.cat([mean_pool, max_pool], dim=1)  # (num_datasets, 512)
        
        dataset_logits = self.classifier(segment_representation)  # (num_datasets, num_classes)

        loss = None
        if self.training:
            if dataset_labels is not None:
                loss = self.criterion(dataset_logits, dataset_labels)
            else:
                raise ValueError("dataset_labels is None in training mode")
        else:
            loss = torch.tensor(0.0, device=dataset_logits.device)

        return padded_feats, dataset_logits, loss


@MODELS.register_module("PointDetector-v1")
class PointCloudDetector(nn.Module):
    """
    A standalone point cloud dataset detector that classifies which dataset a point cloud segment belongs to.
    """
    def __init__(
        self,
        in_channels=6,
        num_classes=3,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
    ):
        super().__init__()
        self.conditions = conditions
        self.dataset_classifier = DatasetClassifier(in_channels=in_channels, num_classes=num_classes)

    def forward(self, data_dict):
        """
        Forward pass of the detector.
        
        Args:
            data_dict: Dictionary containing:
                - condition: List of dataset names
                - offset: Tensor of segment end indices
                - feat: Point features
                - coord: Point coordinates
        
        Returns:
            Dictionary containing:
                - loss: Training loss if in training mode
                - cls_acc: Classification accuracy
                - dataset_logits: Raw classification logits
        """
        condition_list = data_dict["condition"]
        offsets = data_dict["offset"]
        
        dataset_labels = torch.tensor(
            [self.conditions.index(cond) if cond in self.conditions else -1 for cond in condition_list],
            device=data_dict["coord"].device,
        )

        _, dataset_logits, cls_loss = self.dataset_classifier(
            data_dict, offsets, dataset_labels
        )

        # Compute classification accuracy (only for non-ignored labels)
        pred_class = dataset_logits.argmax(dim=1)
        valid_mask = dataset_labels != -1
        if valid_mask.any():
            cls_acc = (pred_class[valid_mask] == dataset_labels[valid_mask]).float().mean()
        else:
            cls_acc = torch.tensor(0.0, device=pred_class.device)

        if self.training:
            return dict(loss=cls_loss, cls_acc=cls_acc)

        pred_condition = [self.conditions[pred_class.cpu().numpy()[i]] for i in range(len(pred_class))]
        return dict(
            pred_condition=pred_condition,
            dataset_logits=dataset_logits,
            cls_acc=cls_acc,
        ) 