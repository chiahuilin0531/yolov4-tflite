import sys
import cv2
import os
import numpy as np
import random
import tensorflow as tf
global_random_seed = 10 # random.randint(0, 65535)
operand_random_seed = 100 # random.randint(0, 65535)
tf.keras.utils.set_random_seed(global_random_seed)
# # tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(global_random_seed)
# np.random.seed(global_random_seed)
# random.seed(global_random_seed)

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

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('draw_image', False, 'draw image or not in visualize_anno directory')

tf.config.optimizer.set_jit(True)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def main(_argv):
    init_shared_variable()
    folder='augmentation'
    os.makedirs('visualize_anno', exist_ok=True)
    os.makedirs(f'visualize_anno/{folder}', exist_ok=True)
    
    filtered_full_HD_area = 123
    filtered_area = filtered_full_HD_area / (1920 / cfg.TRAIN.INPUT_SIZE) ** 2
    filtered_area = max(filtered_area, 20)
    print('[Info] Filtered Instance Area', filtered_area)
    
    trainDataLoader = tfDataset(FLAGS, cfg, is_training=True, filter_area=filtered_area, use_imgaug=False, operand_seed=operand_random_seed)
    batch_size  = trainDataLoader.batch_size 

    trainDataLoader.reset_epoch()
    trainset = trainDataLoader.dataset_gen()
    testset = tfDataset(FLAGS, cfg, is_training=False, filter_area=filtered_area, use_imgaug=False).dataset_gen()
    no_gt_image_cnt = 0
    
    ranges=[123, 195, 267, 341, 489, 637, 786, 1e5]
    bboxes_counter = AreaCounter(ranges)

    with tqdm(total=len(trainset), ncols=150, desc=f"{'Augmenting':<13}") as pbar:
        for batch_idx, data_item in enumerate(trainset):
            if not isinstance(trainset, Dataset):
                # (b,h,w,3)
                batch_image=(data_item['images'].numpy() * 255.0).astype(np.uint8)
                # (b,max_bboxes*2,5)
                batch_bboxes=np.concatenate([data_item['bboxes_m'], data_item['bboxes_l']], axis=1).astype(np.int32)
            else:
                batch_image=(data_item[0] * 255.0).astype(np.uint8)
                batch_bboxes=np.concatenate([data_item[1][0][1], data_item[1][1][1]], axis=1).astype(np.int32)
            processed_img=[]
            
            stats_bboxes = tf.concat([data_item['bboxes_m'], data_item['bboxes_l']], axis=0)
            stats_bboxes = stats_bboxes.numpy()
            bboxes_counter.update(stats_bboxes) 
            
            for i in range(4):
                break
                img=batch_image[i]
                bboxes=batch_bboxes[i]

                illegal_bbox_mask=np.all(bboxes[...,:4] == 0, axis=-1)
                valid_bbox_mask=~illegal_bbox_mask
                valid_bbox=bboxes[valid_bbox_mask]
                
                cnt=0
                min_area=np.inf
                

                
                for bbox in valid_bbox:
                    cnt+=1
                    print('area', bbox[2]*bbox[3],'bbox', bbox)
                    min_area = min(bbox[2]*bbox[3], min_area)
                    top_left = tuple(map(int, bbox[:2] - bbox[2:4] // 2))
                    bot_right = tuple(map(int, bbox[:2] + bbox[2:4] // 2))
                    img = np.ascontiguousarray(img)
                    img = cv2.rectangle(img, top_left, bot_right, (255,0,0), 1)
                if cnt == 0: 
                    print(f'non of bboxes in {batch_idx} {i} img')
                    no_gt_image_cnt+=1
                # img = cv2.putText(img, f'num of box {cnt} min box area: {min_area}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                
                single_img_bboxes = data_item['bboxes_m'][i]
                single_area = (single_img_bboxes[:,2] - single_img_bboxes[:,0]) * (single_img_bboxes[:,3] - single_img_bboxes[:,1])
                single_area = single_area[single_area > 1]
                img = cv2.putText(img, f'num of box {cnt} min area: {min_area}. cnt: {single_area.shape[0]}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                
                processed_img.append(img)
            
            # top_img=np.concatenate(processed_img[:2], axis=1)
            # bot_img=np.concatenate(processed_img[2:], axis=1)
            # full_img = np.concatenate([top_img, bot_img], axis=0)
            # cv2.imwrite(os.path.join('visualize_anno',folder, f'{batch_idx}.jpg'), full_img[...,::-1])
            
            pbar.set_postfix({
                "no_gt_image_cnt": f"{no_gt_image_cnt:5d}"
            })
            pbar.update(1)

        print('\n', bboxes_counter.get_info())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
