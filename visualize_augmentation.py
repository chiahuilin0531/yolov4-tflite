from builtins import isinstance
from core.common import MishLayer,BatchNormalization
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator
import os, shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset_tiny import Dataset, tfDataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import draw_bbox
import cv2
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
import time

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
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
    folder='origin'
    os.makedirs('visualize_anno', exist_ok=True)
    os.makedirs(f'visualize_anno/{folder}', exist_ok=True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # trainset = Dataset(FLAGS, is_training=True, filter_area=123)
    # testset = Dataset(FLAGS, is_training=False, filter_area=123)
    #########################################################
    # trainset = tfDataset(FLAGS, is_training=True, filter_area=123).dataset_gen()
    trainset = Dataset(FLAGS, is_training=True, filter_area=123)

    
    for batch_idx, data_item in enumerate(trainset):
        print(batch_idx)
        if not isinstance(trainset, Dataset):
            # (b,h,w,3)
            batch_image=(data_item[0].numpy() * 255.0).astype(np.uint8)
            # (b,max_bboxes*2,5)
            batch_bboxes=np.concatenate([data_item[2], data_item[4]], axis=1).astype(np.int32)
        else:
            batch_image=(data_item[0] * 255.0).astype(np.uint8)
            batch_bboxes=np.concatenate([data_item[1][0][1], data_item[1][1][1]], axis=1).astype(np.int32)
        processed_img=[]
        for i in range(4):
            img=batch_image[i]
            bboxes=batch_bboxes[i]
            bboxes=np.concatenate([bboxes[...,:2] - bboxes[...,2:4]//2, bboxes[...,:2] + bboxes[...,2:4]//2], axis=-1)
            illegal_bbox_mask=np.all(bboxes[...,:] == 0, axis=-1)
            valid_bbox_mask=~illegal_bbox_mask
            valid_bbox=bboxes[valid_bbox_mask]
            
            cnt=0
            for bbox in valid_bbox:
                if bbox[0] == 0 and bbox[1] == 0 and bbox[2] ==0 and bbox[3] == 0:
                    continue
                cnt+=1
                cv2.rectangle(img, bbox[:2], bbox[2:4], (255,0,0), 1)
            img = cv2.putText(img, f'num of box {cnt}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            processed_img.append(img)
        
        top_img=np.concatenate(processed_img[:2], axis=1)
        bot_img=np.concatenate(processed_img[2:], axis=1)
        full_img = np.concatenate([top_img, bot_img], axis=0)

        cv2.imwrite(os.path.join('visualize_anno',folder, f'{batch_idx}.jpg'), full_img[...,::-1])
        if batch_idx == 800: break


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
