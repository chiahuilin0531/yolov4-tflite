import sys
import cv2
import os
import numpy as np
import random
import tensorflow as tf
global_random_seed = random.randint(0, 65535) # 10
operand_random_seed = random.randint(0, 65535) # 100
tf.keras.utils.set_random_seed(global_random_seed)
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator, AreaCounter
import os, shutil
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.iayolo import CNNPP, DIP_FilterGraph

from core.dataset_tiny import Dataset, tfDataset
from core.config import cfg


from core import utils
from core.utils import draw_bbox, freeze_all, unfreeze_all, read_class_names, get_shared_variable, init_shared_variable
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
import tensorflow_addons as tfa
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('qat', False, 'train w/ or w/o quatize aware')

def kmeans_anchor(data, filename):
    n_clusters=3
    kmeans3 = KMeans(n_clusters=n_clusters)
    kmeans3.fit(data)
    y_kmeans3 = kmeans3.predict(data)

    yolo_anchor_average=[]
    for ind in range (n_clusters):
        yolo_anchor_average.append(np.mean(data[y_kmeans3==ind],axis=0))

    yolo_anchor_average=np.array(yolo_anchor_average, dtype=np.int32)

    print(f'anchors are : {yolo_anchor_average}')
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans3, s=2, cmap='viridis')
    plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='black', s=30)
    plt.title(f'yolov3 anchors k-means {n_clusters} clusters')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.savefig(f'{filename}', dpi = 300)

def main(_argv):
    print(f'Global Seed: {global_random_seed} Operation Seed: {operand_random_seed}')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    init_shared_variable()
    filtered_full_HD_area = 123
    filtered_area = filtered_full_HD_area / (1920 / cfg.TRAIN.INPUT_SIZE) ** 2
    print('[Info] Filtered Instance Area', filtered_area)
    cfg.TRAIN.DATA_AUG = False
    cfg.TRAIN.ANNOT_PATHS          = [
        "datasets/data_selection_mix/anno/train_3cls_filter_small.txt",
        "datasets/Taiwan_trafficlight.v1.coco/anno/train_3cls.txt",
        "datasets/night_dataset/anno/train_3cls.txt"    
    ]
    trainset = tfDataset(FLAGS, cfg, is_training=True, filter_area=filtered_area, use_imgaug=False).dataset_gen()

    bboxes_list = []
    for data_item in tqdm(trainset):
        bboxes = tf.concat([data_item['bboxes_m'], data_item['bboxes_l']], axis=1).numpy()
        bboxes = np.reshape(bboxes, (-1, 4))
        area = bboxes[:, 2] * bboxes[:, 3]
        bboxes = bboxes[area > 1]
        
        # print('area', area.mean(), area.max(), area.min())
        
        bboxes_list.append(bboxes)
        
        
    bboxes = np.concatenate(bboxes_list, axis=0)
    print(f'Total Number of BBox {len(bboxes)}')
    print('bboxes', bboxes.shape)
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    area = w * h
    
    print('Kmeans for small bboxes')
    small_w = w[area <= 78].reshape(-1, 1)
    small_h = h[area <= 78].reshape(-1, 1)
    small_data = np.concatenate([small_w, small_h], axis=-1)
    print(small_data.shape)
    kmeans_anchor(small_data, 'anchor_small.png')



    print('Kmeans for large bboxes')
    large_w = w[area > 78].reshape(-1, 1)
    large_h = h[area > 78].reshape(-1, 1)
    large_data = np.concatenate([large_w, large_h], axis=-1)
    kmeans_anchor(large_data, 'anchor_large.png')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
