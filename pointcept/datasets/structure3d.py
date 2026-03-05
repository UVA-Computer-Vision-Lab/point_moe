"""
Structured3D Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import random
from collections.abc import Sequence

from .defaults import DefaultDataset
from .builder import DATASETS

from pointcept.utils.logger import get_root_logger
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        sample_percent=0.1,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.sample_percent = sample_percent
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(
                os.path.join(self.data_root, self.split, "scene_*/room_*")
            )
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.data_root, split, "scene_*/room_*")
                )
        else:
            raise NotImplementedError
        data_list = random.sample(data_list, int(len(data_list) * self.sample_percent))
        return data_list

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, room_name = os.path.split(file_path)
        scene_name = os.path.basename(dir_path)
        data_name = f"{scene_name}_{room_name}"
        return data_name
