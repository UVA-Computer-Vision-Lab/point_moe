# Point-MoE: Large-Scale Multi-Dataset Training with Mixture-of-Experts for 3D Semantic Segmentation

[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://point-moe.cs.virginia.edu/)
[![Paper](https://img.shields.io/badge/arXiv-2505.23926-red.svg)](https://arxiv.org/abs/2505.23926)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/forum?id=35HahPHrFG)

**[Xuweiyi Chen](https://xuweiyichen.github.io/)<sup>1</sup>, [Wentao Zhou](https://smirkkkk.github.io/)<sup>1</sup>, [Aruni RoyChowdhury](https://arunirc.github.io/)<sup>2</sup>, [Zezhou Cheng](https://sites.google.com/site/zezhoucheng/)<sup>1</sup>**

<sup>1</sup>University of Virginia &nbsp; <sup>2</sup>The MathWorks, Inc.

This repository provides the official implementation for the paper [Point-MoE: Large-Scale Multi-Dataset Training with Mixture-of-Experts for 3D Semantic Segmentation](https://arxiv.org/abs/2505.23926), published at **ICLR 2026**.

If you find this code useful, please consider citing:
```bibtex
@inproceedings{chenpoint,
  title={Point-MoE: Large-Scale Multi-Dataset Training with Mixture-of-Experts for 3D Semantic Segmentation},
  author={Chen, Xuweiyi and Zhou, Wentao and RoyChowdhury, Aruni and Cheng, Zezhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```
conda create -n point_moe python=3.11 -y
conda activate point_moe

pip install ninja h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric
pip install spconv-cu118

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```

### Ensuring Compatibility: CUDA, Torch, and ABI Matching

When installing `flash_attn`, **all components must match**, including:
- CUDA Version
- Torch Version
- C++ ABI Compatibility (`cxx11abiFALSE` vs. `cxx11abiTRUE`)

Mismatches can lead to undefined symbol errors or runtime crashes.  
Always verify that the installed `flash_attn` version aligns with your CUDA, PyTorch, and ABI settings.

To avoid waiting, you can try
```
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Datasets
--------

We use preprocessed datasets provided by [Pointcept](https://github.com/Pointcept/Pointcept) via their HuggingFace repository. Please download the datasets from:

https://huggingface.co/Pointcept

Follow their instructions for each dataset to place the data under the `data/` directory.

**Training datasets:** ScanNet, Structured3D, S3DIS, nuScenes, SemanticKITTI

**Zero-shot evaluation datasets:** Matterport3D, Waymo

Train Point-MoE
----------------------

```
sh scripts/train.sh -d point_moe -c indoor -g 4
```

Inference
---------

```
sh scripts/test.sh -n results/indoor -g 1
```

> **Note:** Training was done on 4 A100 GPUs for the indoor-only model and 8 A100 GPUs for the indoor+outdoor model.

TODO
----

- [ ] Release pretrained checkpoints

Acknowledgements
----------------

We would like to thank [Pointcept](https://github.com/Pointcept/Pointcept) for open-sourcing their codebase and preprocessed datasets, which form the foundation of this work. We plan to integrate Point-MoE into the Pointcept framework in the future if possible.

