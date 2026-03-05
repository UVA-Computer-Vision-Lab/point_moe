"""
Evaluate Hook

Author: Xuweiyi Chen (xuweic@email.virginia.edu)
        Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4
from tqdm import tqdm
from datetime import datetime

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS
import wandb
import numpy as np
from collections import defaultdict
from pointcept.models.point_moe.moe_layer import MoELayer, MoEMLP, MoEGate
from pointcept.models.point_moe.point_moe_v0m0_base import PointTransformerMoE, SerializedPooling, SerializedUnpooling
from pointcept.models.utils.structure import Point
import os
import pickle
import torch.nn.functional as F
import tracemalloc
import gc
import pandas as pd
import psutil
import sys
import time

@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, moe_hook_enabled=True):
        self.moe_hook_enabled = moe_hook_enabled
        self.moe_layers = None

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        if self.moe_hook_enabled:
            self.moe_layers = register_moe_hooks(self.trainer.model)
        self.trainer.model.eval()
        
        # Reset MoE counters if enabled
        if self.moe_hook_enabled:
            for module in self.moe_layers.values():
                module.expert_token_counter.zero_()
                module.gate.score_accumulator.zero_()
                
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
            
        # Get MoE token distribution if enabled
        if self.moe_hook_enabled:
            moe_token_distribution = {
                name: module.expert_token_counter.cpu().tolist()
                for name, module in self.moe_layers.items()
            }
            # Convert to ratios
            moe_token_distribution_ratio = {
                name: [
                    count / sum(counts) if sum(counts) > 0 else 0.0
                    for count in counts
                ]
                for name, counts in moe_token_distribution.items()
            }
            moe_gate_scores = {
                name: module.gate.score_accumulator.tolist()
                for name, module in self.moe_layers.items()
            }
            
            # Log MoE distribution
            self.trainer.logger.info("MoE Token Distribution (Ratio):")
            for moe_name, expert_ratios in moe_token_distribution_ratio.items():
                ratio_str = ", ".join([f"{r:.2%}" for r in expert_ratios])
                self.trainer.logger.info(f" - {moe_name}: [{ratio_str}]")
            self.trainer.logger.info("MoE Gate Scores:")
            for moe_name, gate_scores in moe_gate_scores.items():
                scores_str = ", ".join([f"{s:.3f}" for s in gate_scores])
                self.trainer.logger.info(f" - {moe_name}: [{scores_str}]")
            # Save to file
            save_dict = {
                "moe_token_distribution": moe_token_distribution,
                "moe_token_distribution_ratio": moe_token_distribution_ratio,
                "moe_gate_scores": moe_gate_scores
            }
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.trainer.cfg.save_path, f"moe_token_distribution_{timestamp}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(save_dict, f)
                
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1, freq=10):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.freq = freq
        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.epoch % self.freq == 0:
            if self.trainer.cfg.evaluate:
                self.eval()

    def associate_instances(self, pred, segment, instance, dataset_name):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(len(self.trainer.cfg.data.names)):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        # breakpoint()
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            # breakpoint()
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for dataset_name, dataset_loader in self.trainer.val_loader.items():
            scenes = []
            valid_index = self.trainer.cfg.model.backbone.conditions.index(dataset_name)
            valid_index = self.trainer.cfg.model.backbone.valid_index[valid_index]
            valid_class_names = [self.trainer.cfg.model.backbone.class_name[i] for i in valid_index]
            self.trainer.cfg.data.names  = valid_class_names
            self.valid_class_names = valid_class_names
            # breakpoint()
            for i, input_dict in enumerate(dataset_loader):
                print(f"Processing {dataset_name} batch {i}")
                assert (
                    len(input_dict["offset"]) == 1
                )  # currently only support bs 1 for each GPU
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)

                loss = output_dict["loss"]

                segment = input_dict["segment"]
                instance = input_dict["instance"]
                # map to origin
                if "origin_coord" in input_dict.keys():
                    idx, _ = pointops.knn_query(
                        1,
                        input_dict["coord"].float(),
                        input_dict["offset"].int(),
                        input_dict["origin_coord"].float(),
                        input_dict["origin_offset"].int(),
                    )
                    idx = idx.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                    segment = input_dict["origin_segment"]
                    instance = input_dict["origin_instance"]

                gt_instances, pred_instance = self.associate_instances(
                    output_dict, segment, instance, dataset_name
                )
                scenes.append(dict(gt=gt_instances, pred=pred_instance))

                self.trainer.storage.put_scalar("val_loss", loss.item())
                self.trainer.logger.info(
                    "Test: [{iter}/{max_iter}] "
                    "Loss {loss:.4f} ".format(
                        iter=i + 1, max_iter=len(dataset_loader), loss=loss.item()
                    )
                )

            loss_avg = self.trainer.storage.history("val_loss").avg
            comm.synchronize()
            scenes_sync = comm.gather(scenes, dst=0)
            scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
            ap_scores = self.evaluate_matches(scenes)
            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            self.trainer.logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            valid_index = self.trainer.cfg.model.backbone.conditions.index(dataset_name)
            valid_index = self.trainer.cfg.model.backbone.valid_index[valid_index]
            valid_class_names = [self.trainer.cfg.model.backbone.class_name[i] for i in valid_index]
            self.valid_class_names = valid_class_names
            for i, label_name in enumerate(valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
                self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
                self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
                self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
            self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
            self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
            self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver

def register_moe_hooks(model):
    name_to_moe = {}
    
    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            name_to_moe[name] = module
            module.layer_name = name

            def make_hook(m):
                def hook(_, input, output):
                    # breakpoint()
                    with torch.no_grad():
                        topk_idx = m.gate(input[0].feat)[0]
                        flat_idx = topk_idx.view(-1)
                        for i in range(m.expert_token_counter.numel()):
                            m.expert_token_counter[i] += (flat_idx == i).sum().item()
                return hook

            module.register_forward_hook(make_hook(module))

        elif isinstance(module, MoEGate):
            def make_gate_hook(m):
                def hook(_, input, output):
                    with torch.no_grad():
                        if isinstance(input[0], Point):
                            intermediate_feat = input[0].feat
                        else:
                            intermediate_feat = input[0]
                        logits = F.linear(intermediate_feat, m.weight)
                        scores = logits.softmax(dim=-1)
                        m.score_accumulator.copy_(scores.mean(dim=0).cpu())
                return hook

            module.register_forward_hook(make_gate_hook(module))

    return name_to_moe

def register_feature_hooks(model):
    features = defaultdict(list)
    
    # Register hook for encoder
    def make_enc_hook():
        def hook(_, input, output):
            with torch.no_grad():
                # Get the point features after encoder
                enc_feat = output.feat
                # Pool features to a single vector using mean
                pooled_enc = torch.mean(enc_feat, dim=0)
                features["encoder"].append(pooled_enc.cpu().numpy())
        return hook
    
    # Register hook for decoder
    def make_dec_hook():
        def hook(_, input, output):
            with torch.no_grad():
                # Get the point features after decoder
                dec_feat = output.feat
                # Pool features to a single vector using mean
                pooled_dec = torch.mean(dec_feat, dim=0)
                features["decoder"].append(pooled_dec.cpu().numpy())
        return hook
    
    # Find and register hooks for PointTransformerMoE
    for name, module in model.named_modules():
        if isinstance(module, PointTransformerV3) or isinstance(module, PointTransformerMoE) or isinstance(module, PointTransformerV3MoE):
            # Register hook for encoder
            module.enc.register_forward_hook(make_enc_hook())
            # Register hook for decoder
            module.dec.register_forward_hook(make_dec_hook())
            break  # Only need to register once
    
    return features

def register_moe_score_hooks(model):
    scores = defaultdict(list)

    for name, module in model.named_modules():
        if isinstance(module, MoEGate):
            def make_gate_hook(m):
                def hook(_, input, output):
                    with torch.no_grad():
                        logits = F.linear(input[0], m.weight)
                        if name not in scores:
                            scores[name] = []
                        scores[name].append(logits.softmax(dim=-1).cpu().numpy())
                return hook
            module.register_forward_hook(make_gate_hook(module))
    
    return scores

def register_pooling_unpooling_hooks(model):
    points_pooling = {}
    points_unpooling = defaultdict(list)
    
    for name, module in model.named_modules():
        if isinstance(module, SerializedPooling):
            def make_pooling_hook(m):
                def hook(_, input, output):
                    with torch.no_grad():
                        if name not in points_pooling:
                            points_pooling[name] = []
                        if len(points_pooling[name]) == 0:
                            points_pooling[name].append(input[0]["coord"].cpu().numpy())
                        points_pooling[name].append(output["coord"].cpu().numpy())
                return hook
            module.register_forward_hook(make_pooling_hook(module))
            
        elif isinstance(module, SerializedUnpooling):
            def make_unpooling_hook(m):
                def hook(_, input, output):
                    with torch.no_grad():
                        if name not in points_unpooling:
                            points_unpooling[name] = []
                        # if len(points_unpooling) == 0:
                            # points_unpooling["original"] = input[0]["coord"].cpu().numpy()
                        points_unpooling[name].append(output["coord"].cpu().numpy())
                return hook
            
            module.register_forward_hook(make_unpooling_hook(module))
    
    return points_pooling, points_unpooling
                            

def save_features(features, save_path, dataset_name):
    """Save collected features as numpy arrays."""
    feature_dir = os.path.join(save_path, "features")
    os.makedirs(feature_dir, exist_ok=True)
    
    for layer_name, feature_list in features.items():
        if len(feature_list) > 0:  # Only save if we have features
            # Stack all features for this layer
            stacked_features = np.stack(feature_list)
            # Save as numpy array
            np.save(os.path.join(feature_dir, f"{dataset_name}_{layer_name}.npy"), stacked_features)
            print(f"Saved {len(feature_list)} {layer_name} features for {dataset_name}")

def get_process_memory():
    """Get memory usage information for the current process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    result = {
        # Process memory metrics
        "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "shared_mb": getattr(mem_info, 'shared', 0) / (1024 * 1024),  # Shared memory in MB
        "proc_percent": process.memory_percent(),  # Process memory as % of total
        
        # System memory metrics
        "system_total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "system_available_mb": psutil.virtual_memory().available / (1024 * 1024),
        "system_used_mb": psutil.virtual_memory().used / (1024 * 1024),
        "system_percent": psutil.virtual_memory().percent,
    }
    
    return result

def save_scores(expert_selection_per_token, save_path, dataset_name, batch_idx):
    score_dir = os.path.join(save_path, "scores")
    os.makedirs(score_dir, exist_ok=True)
    
    save_path = os.path.join(score_dir, f"{dataset_name}_scores_{batch_idx}.npy")
    with open(save_path, "wb") as f:
        pickle.dump(expert_selection_per_token, f)
    print(f"Saved {len(expert_selection_per_token)} scores for {dataset_name}")

def save_points(points_pooling, points_unpooling, save_path, dataset_name, batch_idx):
    points_dir = os.path.join(save_path, "points")
    os.makedirs(points_dir, exist_ok=True)
    
    save_pooling_path = os.path.join(points_dir, f"{dataset_name}_points_pooling_{batch_idx}.npy")
    save_unpooling_path = os.path.join(points_dir, f"{dataset_name}_points_unpooling_{batch_idx}.npy")
    with open(save_pooling_path, "wb") as f:
        pickle.dump(points_pooling, f)
    with open(save_unpooling_path, "wb") as f:
        pickle.dump(points_unpooling, f)
    print(f"Saved {len(points_pooling)} points for {dataset_name}")

@HOOKS.register_module()
class DatasetWiseSemSegEvaluator(HookBase):
    def __init__(self, eval_freq=20, moe_hook_enabled=False, save_features=False, profile_memory=False, save_expert_selection_per_token=False):
        self.eval_freq = eval_freq
        self.moe_hook_enabled = moe_hook_enabled
        self.save_features = save_features
        self.profile_memory = profile_memory
        self.features = None
        self.memory_stats = []  # List of memory measurement records
        self.save_expert_selection_per_token = save_expert_selection_per_token
        self.num_save_scene = 1
        
    def _log_memory(self, tag, dataset_name=None, batch_idx=None):
        """Log memory usage with contextual information."""
        if not self.profile_memory:
            return
            
        # Force garbage collection before measurement
        gc.collect()
        
        # Get memory info
        memory_info = get_process_memory()
        
        # Add contextual information
        memory_info["tag"] = tag
        memory_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_info["timestamp_seconds"] = time.time()
        
        if dataset_name:
            memory_info["dataset"] = dataset_name
        if batch_idx is not None:
            memory_info["batch_idx"] = batch_idx
            
        # Add to records
        self.memory_stats.append(memory_info)
        
        # Log to console
        self.trainer.logger.info(f"Memory usage - {tag}" + 
                               (f" - {dataset_name}" if dataset_name else "") + 
                               (f" - batch {batch_idx}" if batch_idx is not None else ""))
        self.trainer.logger.info(f"  Process: {memory_info['rss_mb']:.2f} MB RSS, {memory_info['proc_percent']:.2f}% of system memory")
        self.trainer.logger.info(f"  System: {memory_info['system_used_mb']:.2f}/{memory_info['system_total_mb']:.2f} MB ({memory_info['system_percent']:.1f}% used)")
        
        return memory_info
        
    def before_eval(self):
        """Prepare for evaluation."""
        if self.save_features:
            self.features = register_feature_hooks(self.trainer.model)
        
        if self.profile_memory:
            # Reset memory stats for this evaluation
            self.memory_stats = []
            # Initial memory measurement
            self._log_memory("start_evaluation")

        if self.save_expert_selection_per_token:
            self.expert_selection_per_token = register_moe_score_hooks(self.trainer.model)
            self.points_pooling, self.points_unpooling = register_pooling_unpooling_hooks(self.trainer.model)
        
        # Initialize hooks and trackers
        if self.moe_hook_enabled:
            self.moe_layers = register_moe_hooks(self.trainer.model)

    def after_eval(self):
        """Clean up after evaluation."""
        if self.save_features:
            self.features = None
        
        if self.save_expert_selection_per_token:
            self.expert_selection_per_token = None
            self.points_pooling, self.points_unpooling = None, None
            
        if self.profile_memory:
            # Final memory measurement
            self._log_memory("end_evaluation")
            
            # Save all memory statistics to CSV
            if self.memory_stats:
                memory_df = pd.DataFrame(self.memory_stats)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                memory_file = os.path.join(self.trainer.cfg.save_path, f"memory_profile_{timestamp}.csv")
                memory_df.to_csv(memory_file, index=False)
                self.trainer.logger.info(f"Memory profiling data saved to {memory_file}")
                
                # Create memory summary
                summary = {}
                
                # Get tags with multiple entries for statistics
                tags = set()
                datasets = set()
                for record in self.memory_stats:
                    tags.add(record["tag"])
                    if "dataset" in record:
                        datasets.add(record["dataset"])
                
                # Calculate max memory per tag
                for tag in tags:
                    tag_records = [r for r in self.memory_stats if r["tag"] == tag]
                    if tag_records:
                        max_rss = max(r["rss_mb"] for r in tag_records)
                        summary[f"max_rss_mb_{tag}"] = max_rss
                
                # Calculate max memory per dataset
                for dataset in datasets:
                    dataset_records = [r for r in self.memory_stats if r.get("dataset") == dataset]
                    if dataset_records:
                        max_rss = max(r["rss_mb"] for r in dataset_records)
                        summary[f"max_rss_mb_{dataset}"] = max_rss
                
                # Print memory summary
                self.trainer.logger.info("Memory Usage Summary:")
                for key, value in summary.items():
                    self.trainer.logger.info(f"  {key}: {value:.2f} MB")

    def eval(self):
        if (self.trainer.epoch + 1) % self.eval_freq == 0:
            self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
                
            self.before_eval()
            
            self.trainer.model.eval()

            val_loaders = self.trainer.val_loader
            ignore_index = self.trainer.cfg.data.ignore_index

            dataset_results = defaultdict(lambda: defaultdict(list))
            wandb_metrics = {}
            moe_token_distribution_per_dataset = {}

            for dataset_name, val_loader in tqdm(val_loaders.items(), desc="Evaluating datasets", leave=False):
                self.trainer.logger.info(f"Evaluating {dataset_name}...")
                
                # Memory profile at start of dataset
                if self.profile_memory:
                    self._log_memory("dataset_start", dataset_name)
                
                # Reset MoE layer token counters and features
                if self.moe_hook_enabled:
                    for module in self.moe_layers.values():
                        module.expert_token_counter.zero_()
                        module.gate.score_accumulator.zero_()
                if self.save_features:
                    self.features.clear()
                if self.save_expert_selection_per_token:
                    self.expert_selection_per_token.clear()
                    self.points_pooling.clear()
                    self.points_unpooling.clear()

                # Process each batch
                for i, input_dict in enumerate(tqdm(val_loader, desc=f"Processing {dataset_name}", leave=False)):
                    if self.save_expert_selection_per_token:
                        self.expert_selection_per_token.clear()
                        self.points_pooling.clear()
                        self.points_unpooling.clear()
                    # Memory profile at regular intervals (first batch, every 10th batch)
                    if self.profile_memory and (i == 0 or i % 10 == 0):
                        self._log_memory("batch_processing", dataset_name, i)
                    
                    # Move tensors to device
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    
                    # Memory profile before forward pass for first batch
                    if self.profile_memory and i == 0:
                        self._log_memory("before_forward_pass", dataset_name, i)
                    
                    # Forward pass
                    with torch.no_grad():
                        output_dict = self.trainer.model(input_dict)
                    
                    # Memory profile after forward pass for first batch
                    if self.profile_memory and i == 0:
                        self._log_memory("after_forward_pass", dataset_name, i)

                    # Process outputs and calculate metrics
                    index = self.trainer.cfg.model.conditions.index(
                        input_dict["condition"][0]
                    )
                    selected_class_names = [
                        self.trainer.cfg.model.class_name[i]
                        for i in self.trainer.cfg.model.valid_index[index]
                    ]

                    num_classes = len(selected_class_names)
                    output = output_dict["seg_logits"]
                    loss = output_dict["loss"]

                    pred = (
                        output.max(1)[1]
                        if not isinstance(output, list)
                        else output[0].max(1)[1]
                    )
                    segment = input_dict["segment"]

                    if "origin_coord" in input_dict.keys():
                        idx, _ = pointops.knn_query(
                            1,
                            input_dict["coord"].float(),
                            input_dict["offset"].int(),
                            input_dict["origin_coord"].float(),
                            input_dict["origin_offset"].int(),
                        )
                        pred = pred[idx.flatten().long()]
                        segment = input_dict["origin_segment"]

                    intersection, union, target = intersection_and_union_gpu(
                        pred, segment, num_classes, ignore_index
                    )

                    # Aggregate results across GPUs
                    if dist.is_initialized():
                        dist.all_reduce(intersection)
                        dist.all_reduce(union)
                        dist.all_reduce(target)

                    dataset_results[dataset_name]["index"].append(index)
                    dataset_results[dataset_name]["intersection"].append(
                        intersection.cpu().numpy()
                    )
                    dataset_results[dataset_name]["union"].append(union.cpu().numpy())
                    dataset_results[dataset_name]["target"].append(target.cpu().numpy())
                    dataset_results[dataset_name]["loss"].append(loss.item())
                    
                    if self.save_expert_selection_per_token and i < self.num_save_scene:
                        save_scores(self.expert_selection_per_token, self.trainer.cfg.save_path, dataset_name, i)
                        save_points(self.points_pooling, self.points_unpooling, self.trainer.cfg.save_path, dataset_name, i)
                        breakpoint()
                
                # Memory profile at end of dataset processing
                if self.profile_memory:
                    self._log_memory("dataset_end", dataset_name)
                
                # Collect MoE distribution data if enabled
                if self.moe_hook_enabled:
                    # Store token distribution
                    moe_token_distribution_per_dataset[dataset_name] = {
                        name: module.expert_token_counter.cpu().tolist()
                        for name, module in self.moe_layers.items()
                    }
                    
                    # Store gate scores
                    moe_token_distribution_per_dataset[dataset_name + "_gate_scores"] = {
                        name: module.gate.score_accumulator.tolist()
                        for name, module in self.moe_layers.items()
                    }
                    
                    # Convert to ratios
                    moe_token_distribution_per_dataset[dataset_name + "_ratio"] = {
                        name: [
                            count / sum(counts) if sum(counts) > 0 else 0.0
                            for count in counts
                        ]
                        for name, counts in moe_token_distribution_per_dataset[dataset_name].items()
                    }

                # Compute metrics for this dataset
                intersection = np.sum(
                    np.stack(dataset_results[dataset_name]["intersection"]), axis=0
                )
                union = np.sum(np.stack(dataset_results[dataset_name]["union"]), axis=0)
                target = np.sum(
                    np.stack(dataset_results[dataset_name]["target"]), axis=0
                )

                loss_avg = np.mean(dataset_results[dataset_name]["loss"])
                iou_class = intersection / (union + 1e-10)
                acc_class = intersection / (target + 1e-10)
                m_iou = np.mean(iou_class)
                m_acc = np.mean(acc_class)
                all_acc = sum(intersection) / (sum(target) + 1e-10)

                self.trainer.logger.info(
                    f"[{dataset_name}] Loss: {loss_avg:.4f} | mIoU/mAcc/allAcc {m_iou:.4f}/{m_acc:.4f}/{all_acc:.4f}"
                )

                for i in range(num_classes):
                    self.trainer.logger.info(
                        f"[{dataset_name}] Class_{i}-{selected_class_names[i]}: IoU {iou_class[i]:.4f}, Accuracy {acc_class[i]:.4f}"
                    )

                # Store metrics for logging
                is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
                if is_main_process:
                    wandb_metrics[f"val/loss_{dataset_name}"] = loss_avg
                    wandb_metrics[f"val/mIoU_{dataset_name}"] = m_iou
                    wandb_metrics[f"val/mAcc_{dataset_name}"] = m_acc
                    wandb_metrics[f"val/allAcc_{dataset_name}"] = all_acc
                    
                    # Log MoE distributions if enabled
                    if self.moe_hook_enabled:
                        self.trainer.logger.info(f"[{dataset_name}] MoE Token Distribution (Ratio):")
                        for moe_name, expert_ratios in moe_token_distribution_per_dataset[dataset_name + "_ratio"].items():
                            ratio_str = ", ".join([f"{r:.2%}" for r in expert_ratios])
                            self.trainer.logger.info(f" - {moe_name}: [{ratio_str}]")
                        self.trainer.logger.info(f"[{dataset_name}] MoE Gate Scores:")
                        for moe_name, gate_scores in moe_token_distribution_per_dataset[dataset_name + "_gate_scores"].items():
                            scores_str = ", ".join([f"{s:.3f}" for s in gate_scores])
                            self.trainer.logger.info(f" - {moe_name}: [{scores_str}]")
                    
                    # Save feature data if enabled
                    if self.save_features:
                        save_features(self.features, self.trainer.cfg.save_path, dataset_name)

            # Ensure all processes synchronize before continuing
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            # Log to wandb
            is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
            if is_main_process:
                if wandb is not None and wandb.run is not None:
                    wandb.log(wandb_metrics)
                    print(f"Logged aggregated metrics: {wandb_metrics}")

                if "val/mIoU_ScanNet" in wandb_metrics:
                    self.trainer.comm_info["current_metric_value"] = wandb_metrics[
                        "val/mIoU_ScanNet"
                    ]
                    self.trainer.comm_info["current_metric_name"] = "ScanNet mIoU"
                else:
                    self.trainer.logger.warning(
                        "Note: 'val/mIoU_ScanNet' metric not found in current evaluation."
                    )
                
                # Save MoE distribution data if enabled
                if self.moe_hook_enabled:
                    save_dict_dir = os.path.join(self.trainer.cfg.save_path, f"moe_token_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                    with open(save_dict_dir, "wb") as f:
                        pickle.dump(moe_token_distribution_per_dataset, f)

            self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
            self.after_eval()
                
    def after_epoch(self):
        if self.trainer.cfg.evaluate and (self.trainer.epoch + 1) % self.eval_freq == 0:
            self.eval()

    def after_train(self):
        """Logs the best metric after training."""
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class DatasetClassifierEvaluator(HookBase):
    """Evaluator for dataset classification task."""
    
    def __init__(self, eval_freq=1):
        self.eval_freq = eval_freq

    def after_epoch(self):
        if self.trainer.cfg.evaluate and (self.trainer.epoch + 1) % self.eval_freq == 0:
            self.eval()

    def compute_balanced_accuracy(self, confusion_matrix):
        """Compute balanced accuracy from confusion matrix."""
        # Get per-class accuracy
        per_class_acc = confusion_matrix.diag() / (confusion_matrix.sum(dim=1) + 1e-10)
        # Average across classes
        balanced_acc = per_class_acc.mean()
        return balanced_acc.item()

    def compute_harmonic_mean(self, dataset_results):
        """Compute harmonic mean of per-dataset accuracies."""
        accuracies = [results["accuracy"] for results in dataset_results.values() if "accuracy" in results]
        if not accuracies:
            return 0.0
        # Handle case where all accuracies are 0
        if all(acc == 0.0 for acc in accuracies):
            return 0.0
        return len(accuracies) / sum(1/acc for acc in accuracies if acc > 0)

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        
        val_loaders = self.trainer.val_loader  # Handles multiple datasets separately
        dataset_results = defaultdict(lambda: defaultdict(list))
        wandb_metrics = {}
        
        # Create confusion matrices for both class-wise and dataset-wise predictions
        confusion_matrix = torch.zeros(len(self.trainer.cfg.model.conditions), 
                                    len(self.trainer.cfg.model.conditions))

        # Iterate over each dataset's validation loader
        for dataset_name, val_loader in tqdm(val_loaders.items(), desc="Evaluating datasets", leave=False):
            self.trainer.logger.info(f"Evaluating {dataset_name}...")
            
            total_correct = 0
            total_samples = 0
            total_loss = 0
            prediction_counts = torch.zeros(len(self.trainer.cfg.model.conditions))
            
            # Process batches for current dataset
            for i, input_dict in enumerate(tqdm(val_loader, desc=f"Processing {dataset_name}", leave=False)):
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)
                
                # Get predictions
                dataset_logits = output_dict["dataset_logits"]
                pred_class = dataset_logits.argmax(dim=1)
                
                # Count predictions for all datasets
                for pred in pred_class.cpu():
                    prediction_counts[pred.long()] += 1
                
                total = pred_class.size(0)
                total_samples += total
                
                # Calculate loss if available
                if "loss" in output_dict:
                    total_loss += output_dict["loss"].item() * total
                
                # Get dataset labels
                condition_list = input_dict["condition"]
                dataset_labels = torch.tensor(
                    [self.trainer.cfg.model.conditions.index(cond) if cond in self.trainer.cfg.model.conditions else -1 for cond in condition_list],
                    device=input_dict["coord"].device,
                )
                
                # Update confusion matrix for valid labels
                valid_mask = dataset_labels != -1
                if valid_mask.any():
                    confusion_matrix[dataset_labels[valid_mask].cpu().long(), pred_class[valid_mask].cpu().long()] += 1
                
                # Calculate accuracy for valid labels
                correct = ((pred_class == dataset_labels) & valid_mask).sum().item()
                total_correct += correct

            # Calculate prediction distribution
            prediction_distribution = prediction_counts / total_samples if total_samples > 0 else prediction_counts
            dataset_loss = total_loss / total_samples if total_samples > 0 else 0
            
            # Store results for this dataset
            dataset_results[dataset_name] = {
                "loss": dataset_loss,
                "total": total_samples,
                "prediction_distribution": prediction_distribution
            }
            print(dataset_results)
            # Compute accuracy
            dataset_accuracy = total_correct / total_samples if total_samples > 0 else 0
            dataset_results[dataset_name]["accuracy"] = dataset_accuracy
            dataset_results[dataset_name]["correct"] = total_correct
            
            # Log dataset results
            self.trainer.logger.info(
                f"[{dataset_name}] Loss: {dataset_loss:.4f} | Accuracy: {dataset_accuracy:.4f} ({total_correct}/{total_samples})"
            )
            self.trainer.logger.info(
                f"\n[{dataset_name}] Prediction Distribution:"
            )
            for i, (condition, percentage) in enumerate(zip(self.trainer.cfg.model.conditions, prediction_distribution)):
                self.trainer.logger.info(f"  {condition}: {percentage:.2%}")

        # Calculate overall metrics across all datasets
        total_correct_all = sum(results.get("correct", 0) for results in dataset_results.values())
        total_samples_all = sum(results["total"] for results in dataset_results.values())
        overall_accuracy = total_correct_all / total_samples_all if total_samples_all > 0 else 0
        
        # Calculate balanced accuracy from confusion matrix
        balanced_accuracy = self.compute_balanced_accuracy(confusion_matrix)
        
        # Calculate harmonic mean of per-dataset accuracies
        harmonic_mean = self.compute_harmonic_mean(dataset_results)
        
        # Compute combined metric (weighted average of balanced accuracy and harmonic mean)
        combined_metric = 0.7 * balanced_accuracy + 0.3 * harmonic_mean
        
        # Log overall results
        self.trainer.logger.info(
            f"\nOverall Results:"
            f"\n- Overall Accuracy: {overall_accuracy:.4f} ({total_correct_all}/{total_samples_all})"
            f"\n- Balanced Accuracy: {balanced_accuracy:.4f}"
            f"\n- Harmonic Mean: {harmonic_mean:.4f}"
            f"\n- Combined Metric: {combined_metric:.4f}"
        )

        # Log confusion matrix
        self.trainer.logger.info("\nConfusion Matrix:")
        conditions = self.trainer.cfg.model.conditions
        # Print header
        header = "True\\Pred"
        for cond in conditions:
            header += f"\t{cond[:7]}"  # Truncate long names
        self.trainer.logger.info(header)
        
        # Print each row
        for i, true_class in enumerate(conditions):
            row = f"{true_class[:7]}"  # Truncate long names
            for j in range(len(conditions)):
                row += f"\t{confusion_matrix[i,j]:.0f}"
            self.trainer.logger.info(row)

        # Log to wandb if available
        is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process and wandb.run is not None:
            wandb_metrics = {
                "val/accuracy_overall": overall_accuracy,
                "val/accuracy_balanced": balanced_accuracy,
                "val/accuracy_harmonic": harmonic_mean,
                "val/accuracy_combined": combined_metric,
            }
            # Add per-dataset metrics
            for dataset_name, results in dataset_results.items():
                wandb_metrics[f"val/loss_{dataset_name}"] = results["loss"]
                if "accuracy" in results:
                    wandb_metrics[f"val/accuracy_{dataset_name}"] = results["accuracy"]
                # # Add prediction distribution for all datasets
                # for i, (condition, percentage) in enumerate(zip(self.trainer.cfg.model.conditions, results["prediction_distribution"])):
                #     wandb_metrics[f"val/pred_{dataset_name}_{condition}"] = percentage
            
            wandb.log(wandb_metrics)

        # Save best metric using the combined metric
        self.trainer.comm_info["current_metric_value"] = combined_metric
        self.trainer.comm_info["current_metric_name"] = "Combined Accuracy"
        
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("Combined Accuracy", self.trainer.best_metric_value)
        )
