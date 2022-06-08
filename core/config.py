#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/3cls.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATHS          = [
    # "datasets/night_dataset/anno/train_well_split_1cls.txt",
    # "datasets/data_selection/anno/train_1cls.txt",
    # "datasets/data_selection_2/anno/train_1cls.txt",
    # "datasets/data_selection_3/anno/train_1cls.txt",
    # "datasets/Taiwan_trafficlight.v1.coco/anno/train_1cls.txt"

    "datasets/night_dataset/anno/train_3cls_reduce_06.txt",
    "datasets/data_selection_mix/anno/train_3cls_filter_small.txt",
    # "datasets/Taiwan_trafficlight.v1.coco/anno/train_3cls.txt"
]
__C.TRAIN.ADVERSARIAL_PATHS    = [
    "datasets/night_dataset/anno/train_well_split_1cls.txt",
]

__C.TRAIN.BATCH_SIZE          = 16 # 8 #2
__C.TRAIN.INPUT_SIZE          = 608
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 40
__C.TRAIN.ADVERSERIAL_CONST     = 0.0  # 1, 0.5, 0.05, 0.005, 0000



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATHS           = [
    # "datasets/night_dataset/anno/val_well_split_1cls.txt",
    # "datasets/data_selection/anno/val_1cls.txt",
    # "datasets/data_selection_2/anno/val_1cls.txt",
    # "datasets/data_selection_3/anno/val_1cls.txt"

    "datasets/night_dataset/anno/val_3cls.txt",
    "datasets/data_selection_mix/anno/val_3cls_filter_small.txt",
]
__C.TEST.ADVERSARIAL_PATHS    = [
    "datasets/night_dataset/anno/val_well_split_1cls.txt",
]
__C.TEST.BATCH_SIZE           = 4
__C.TEST.INPUT_SIZE           = 608
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5


