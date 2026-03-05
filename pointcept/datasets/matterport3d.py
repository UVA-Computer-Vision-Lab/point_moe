"""
Matterport3dDataset

Author: Xuweiyi Chen (xuweic@email.virginia.edu)
Please cite our work if the code is helpful to you.
"""

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Matterport3dDataset(DefaultDataset):
    pass
