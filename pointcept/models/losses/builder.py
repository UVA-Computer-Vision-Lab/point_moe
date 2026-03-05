"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry
import torch

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            # if pred.shape != target.shape:
            # #     breakpoint()  # Conditional breakpoint triggers when shape mismatch occurs
            # loss += c(pred, target)
            try:
                if (target == -1).all():
                    print(f"Skipping loss computation as all targets are -1.")
                    return torch.tensor(
                        0.0, device=pred.device, dtype=pred.dtype
                    )  # Return loss as 0 immediately
                loss += c(pred, target)
            except Exception as e:
                print(f"Error computing loss: {e}")
                print(
                    f"Shape mismatch detected: pred.shape={pred.shape}, target.shape={target.shape}"
                )
                import pdb

                pdb.set_trace()  # Pauses execution for debugging
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
