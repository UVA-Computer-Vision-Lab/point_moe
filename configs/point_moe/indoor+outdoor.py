_base_ = ["../_base_/default_runtime.py"]

is_train = True
batch_size = 16  # bs: total bs in all gpus
num_worker = 12
mix_prob = 0
empty_cache = True
enable_amp = True
find_unused_parameters = True
project_name = "PointMoE-v0m0"
experiment_name = "multiple-dataset-training-8-3"
batch_from_single_dataset = False
clip_grad = 1.0
eval_epoch = 240
# trainer
train = dict(
    type="MultiDatasetTrainer",
)
# model settings
model = dict(
    type="PPTMoE-v0m0",
    backbone=dict(
        type="PTMoE-v0m0",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        # MoE parameters
        use_moe=True,  # Master switch for MoE
        use_moe_mlp=False,  # Apply MoE to MLP blocks
        use_moe_proj=True,  # Apply MoE to attention projection
        num_experts=8,  # Number of experts
        topk=2,  # Top-k routing
        moe_use_residual=False,  # Use residual connections around MoE
        # Apply MoE only to deeper layers for efficiency
        moe_layers=None,  # Example indices of layers to apply MoE
        use_ln=False,
        enable_enc=True,
        enable_dec=True,
        aux_loss_alpha=0,
        n_intermediate_size=2,
        # Activation function parameters
        act_fn="relu",  # Options: "relu", "gelu", "silu", "leaky_relu", "elu", "mish"
        act_fn_params=None,  # Additional parameters for activation function
        use_moe_attn=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,
    context_channels=256,
    conditions=(
        "Structured3D",
        "ScanNet",
        "S3DIS",
        "Matterport3D",
        "SemanticKITTI",
        "nuScenes",
        "Waymo",
    ),
    template="[x]",
    clip_model="ViT-B/16",
    # fmt: off
    class_name=(
        # Structured3D, ScanNet, S3DIS, Matterport3D
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
        "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
        "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
        "clutter", "otherstructure", "otherfurniture", "otherprop", "other",
        
        # SemanticKITTI
        "car", "bicycle", "motorcycle", "truck", "other vehicle",
        "person", "person who rides a bicycle", "person who rides a motorcycle", "road", "parking",
        "path for pedestrians at the side of a road", "other ground", "building", "fence", "vegetation",
        "trunk", "terrain", "pole", "traffic sign",
        
        # nuScenes
        "barrier", "bicycle", "bus", "car", "construction vehicle",
        "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
        "path suitable or safe for driving", "other flat", "sidewalk", "terrain", "man made", "vegetation",
        
        # Waymo
        "car", "truck", "bus", "other vehicle", "person who rides a motorcycle",
        "person who rides a bicycle", "pedestrian", "sign", "traffic light", "pole",
        "construction cone", "bicycle", "motorcycle", "building", "vegetation",
        "tree trunk", "curb", "road", "lane marker", "other ground", "horizontal surface that can not drive",
        "surface when pedestrians most likely to walk on",
    ),
    valid_index=(
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
        (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 36, 19),
        # SemanticKITTI (starts at index 37)
        tuple(range(37, 37 + 19)),

        # nuScenes (starts after SemanticKITTI, so at index 37 + 19 = 56)
        tuple(range(56, 56 + 16)),

        # Waymo (starts after nuScenes, so at index 56 + 16 = 72)
        tuple(range(72, 72 + 22)),
    ),
    # fmt: on
    backbone_mode=False,
)

# scheduler settings
epoch = 240
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.05)

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            # Structured3D
            dict(
                type="Structured3DDataset",
                split="train",
                data_root="data/structured3d",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.8, mode="random"),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "Structured3D"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=1,  # sampling weight
                sample_percent=1,
            ),
            # ScanNet
            dict(
                type="ScanNetDataset",
                split="train",
                data_root="data/scannet",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "ScanNet"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=1,  # sampling weight
            ),
            # S3DIS
            dict(
                type="S3DISDataset",
                split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
                data_root="data/s3dis",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.6, mode="random"),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ShufflePoint"),
                    dict(type="Add", keys_dict={"condition": "S3DIS"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=1,  # sampling weight
            ),
            dict(
                type="SemanticKITTIDataset",
                split="train",
                data_root="data/semantic_kitti",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "strength", "segment"),
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "SemanticKITTI"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
            dict(
                type="NuScenesDataset",
                split="train",
                data_root="data/nuscenes",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='x', p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='y', p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "strength", "segment"),
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "nuScenes"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
        ],
    ),
    val=dict(
        type="MultiValDataset",
        datasets=[
            dict(
                type="ScanNetDataset",
                split="val",
                data_root="data/scannet",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(type="Add", keys_dict={"condition": "ScanNet"}),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
            ),
            dict(
                type="Structured3DDataset",
                split="val",
                data_root="data/structured3d",
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="Add", keys_dict={"condition": "Structured3D"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
            ),
            # dict(
            #     type="S3DISDataset",
            #     split="Area_5",
            #     data_root="data/s3dis",
            #     transform=[
            #         dict(type="CenterShift", apply_z=True),
            #         dict(
            #             type="Copy",
            #             keys_dict={
            #                 "coord": "origin_coord",
            #                 "segment": "origin_segment",
            #             },
            #         ),
            #         dict(
            #             type="GridSample",
            #             grid_size=0.02,
            #             hash_type="fnv",
            #             mode="train",
            #             return_grid_coord=True,
            #         ),
            #         dict(type="CenterShift", apply_z=False),
            #         dict(type="NormalizeColor"),
            #         dict(type="ToTensor"),
            #         dict(type="Add", keys_dict={"condition": "S3DIS"}),
            #         dict(
            #             type="Collect",
            #             keys=(
            #                 "coord",
            #                 "grid_coord",
            #                 "origin_coord",
            #                 "segment",
            #                 "origin_segment",
            #                 "condition",
            #             ),
            #             offset_keys_dict=dict(
            #                 offset="coord", origin_offset="origin_coord"
            #             ),
            #             feat_keys=("color", "normal"),
            #         ),
            #     ],
            #     test_mode=False,
            # ),
            # dict(
            #     type="Matterport3dDataset",
            #     split="val",
            #     data_root="data/matterport3d",
            #     transform=[
            #         dict(type="CenterShift", apply_z=True),
            #         dict(
            #             type="GridSample",
            #             grid_size=0.02,
            #             hash_type="fnv",
            #             mode="train",
            #             return_grid_coord=True,
            #         ),
            #         dict(type="CenterShift", apply_z=False),
            #         dict(type="NormalizeColor"),
            #         dict(type="ToTensor"),
            #         dict(type="Add", keys_dict={"condition": "Matterport3D"}),
            #         dict(
            #             type="Collect",
            #             keys=("coord", "grid_coord", "segment", "condition"),
            #             feat_keys=("color", "normal"),
            #         ),
            #     ],
            #     test_mode=False,
            # ),
        ],
    ),
    # test=dict(
    #     type="MultiDatasetTester",
    #     datasets=[
    #     ],
    # ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="DatasetWiseSemSegEvaluator", eval_freq=10),
    dict(type="CheckpointSaver", save_freq=10),
    dict(type="PreciseEvaluator", test_last=False),
]