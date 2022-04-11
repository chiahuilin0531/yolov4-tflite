import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
##################################
import time
from imgaug import augmenters as iaa
import imgaug as ia
import tensorflow_addons as tfa

class Dataset(object):
    def __init__(self, FLAGS, is_training: bool, dataset_type: str = "converted_coco", filter_area=123):
        self.tiny = FLAGS.tiny
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type

        self.annot_paths = (
            cfg.TRAIN.ANNOT_PATHS if is_training else cfg.TEST.ANNOT_PATHS
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.filter_area = filter_area

    def load_annotations(self):
        annotations = []
        for annot_path in self.annot_paths:
            with open(annot_path, "r") as f:
                txt = f.readlines()
                if self.dataset_type == "converted_coco":
                    annotations += [
                        line.strip()
                        for line in txt
                        if len(line.strip().split()[1:]) != 0
                    ]
                elif self.dataset_type == "yolo":
                    for line in txt:
                        image_path = line.strip()
                        root, _ = os.path.splitext(image_path)
                        with open(root + ".txt") as fd:
                            boxes = fd.readlines()
                            string = ""
                            for box in boxes:
                                box = box.strip()
                                box = box.split()
                                class_num = int(box[0])
                                center_x = float(box[1])
                                center_y = float(box[2])
                                half_width = float(box[3]) / 2
                                half_height = float(box[4]) / 2
                                string += " {},{},{},{},{}".format(
                                    center_x - half_width,
                                    center_y - half_height,
                                    center_x + half_width,
                                    center_y + half_height,
                                    class_num,
                                )
                            annotations.append(image_path + string)
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            # self.train_input_size = random.choice(self.train_input_sizes)
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )
            
            batch_label_bboxes = []       
            batch_bboxes = []             
            for size in self.train_output_sizes:
                label_bbox = np.zeros((self.batch_size, size, size, self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
                batch_label_bboxes.append(label_bbox)
                batch_bbox = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
                batch_bboxes.append(batch_bbox)
            
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    if bboxes.shape[0] == 0: 
                        del self.annotations[index]
                        self.num_samples = len(self.annotations)
                        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
                        continue
                    (
                        label_bboxes,
                        bboxes,
                    )  = self.preprocess_true_boxes(bboxes, num_anchors = len(self.train_output_sizes))
                    # label_bboxes=[(h,w,num_of_anchor_per_scale,6)...... num_of_fpn]
                    # bboxes=[(num_of_max_bboxes, 4)...... num_of_fpn]
                    # label_bboxes, bboxes = preprocess_true_boxes_jit(
                    #     bboxes, 
                    #     len(self.train_output_sizes), 
                    #     self.train_output_sizes, 
                    #     self.anchor_per_scale,
                    #     self.num_classes,
                    #     self.max_bbox_per_scale,
                    #     self.strides,
                    #     self.anchors
                    # )
                    
                    batch_image[num, :, :, :] = image
                    for batch_bbox, bbox in zip(batch_bboxes, bboxes):
                        batch_bbox[num,:,:] = bbox
                    for batch_label_bbox, label_bbox in zip (batch_label_bboxes, label_bboxes):
                        batch_label_bbox[num, :, :, :] = label_bbox
                    num += 1
                self.batch_count += 1
                batch_targets = list(zip(batch_label_bboxes, batch_bboxes))

                return (
                    batch_image,
                    batch_targets
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        if self.dataset_type == "converted_coco":
            bboxes = np.array(
                [list(map(int, box.split(","))) for box in line[1:]]
            )
        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = np.array(
                [list(map(float, box.split(","))) for box in line[1:]]
            )
            bboxes = bboxes * np.array([width, height, width, height, 1])
            bboxes = bboxes.astype(np.int64)

        # Discard too small bounding box
        area = (bboxes[:,0] - bboxes[:,2]) * (bboxes[:,1] - bboxes[:,3])
        mask = area > self.filter_area
        bboxes = bboxes[mask]

        if self.data_aug and bboxes.shape[0] != 0:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        return image, bboxes

    def preprocess_true_boxes(self, bboxes, num_anchors=3):
        """
        Generate dense anchor with ground truth
        Parameter
        ----------
        bboxes: np.float32(num_of_bboxes, 5)\
            the ground truth bbox including x1y1x2y2 class
        num_anchor: list of int\
            the stride for downsample image to FPN size
        Return 
        ------
        list of label
        list of bboxes_xywh
        """
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(num_anchors)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(num_anchors)]
        bbox_count = np.zeros((num_anchors,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(num_anchors):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        #label_sbbox, label_mbbox, label_lbbox = label
        #sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label, bboxes_xywh #label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs



class tfDataset(object):
    def __init__(self, FLAGS, is_training: bool, dataset_type: str = "converted_coco", filter_area=123, use_imgaug=True):
        self.tiny = FLAGS.tiny
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type

        self.annot_paths = (
            cfg.TRAIN.ANNOT_PATHS if is_training else cfg.TEST.ANNOT_PATHS
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 30

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.filter_area = filter_area
        self.create_augment_env()
        self.use_imgaug=use_imgaug

    def load_annotations(self):
        annotations = []
        for annot_path in self.annot_paths:
            with open(annot_path, "r") as f:
                txt = f.readlines()
                if self.dataset_type == "converted_coco":
                    annotations += [
                        line.strip()
                        for line in txt
                        if len(line.strip().split()[1:]) != 0
                    ]
                elif self.dataset_type == "yolo":
                    for line in txt:
                        image_path = line.strip()
                        root, _ = os.path.splitext(image_path)
                        with open(root + ".txt") as fd:
                            boxes = fd.readlines()
                            string = ""
                            for box in boxes:
                                box = box.strip()
                                box = box.split()
                                class_num = int(box[0])
                                center_x = float(box[1])
                                center_y = float(box[2])
                                half_width = float(box[3]) / 2
                                half_height = float(box[4]) / 2
                                string += " {},{},{},{},{}".format(
                                    center_x - half_width,
                                    center_y - half_height,
                                    center_x + half_width,
                                    center_y + half_height,
                                    class_num,
                                )
                            if string=="":
                                continue
                            else:
                                annotations.append(image_path + string)
        np.random.seed(0)
        np.random.shuffle(annotations)
        np.random.shuffle(annotations)
        return annotations

    def create_augment_env(self):
        seq = iaa.Sequential([
            # iaa.Sometimes(p, 
            #     iaa.Affine(
            #         translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            #         # shear={"x": (-10, 10), "y": (-5, 5)},
            #     )
            # ),
            # iaa.Sometimes(0.3, 
            #     iaa.CoarseDropout(p=(0.05, 0.1), size_percent=(0.5,0.25)),
            # ),
            iaa.Sometimes(0.3,
                iaa.OneOf([
                    iaa.Multiply((0.7, 1.2)),
                    iaa.pillike.EnhanceColor(factor=(0.8, 2.0))
                ])
            )
        ], random_order=True)
        self.aug_env = seq

    def dataset_gen(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.annotations)
        if self.data_aug:
            dataset=dataset.shuffle(buffer_size=2048)
        dataset = dataset.map(self.parse_annotation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        

        if self.data_aug:
            dataset = dataset.map(self.random_horizontal_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(self.random_translate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(self.pad_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.pad_bbox, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        if self.data_aug and self.use_imgaug:
            dataset = dataset.map(
                lambda x,y: tf.numpy_function(self.do_augmentation, [x,y], [tf.uint8, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        dataset = dataset.map(
            lambda x,y: tf.numpy_function(self.generate_gth, [x,y], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    @tf.function
    def parse_annotation(self, annotation):
        """
        Parameter
        ---------
        annotation: a string for specify image path and bounding box coordinate

        Return
        ------
        image: uint8(train_size, train_size, 3)\\
            a image with training size 
        bboxes: float32(num_of_max_bbox, 5)\\
            bounding box coordinate(x1y1x2y2) and class
        """
        
        line = tf.strings.split(annotation)
        image_path = line[0]
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        
        # img = tf.cast(tf.image.resize(img, [nh, nw]), dtype=tf.uint8)
        if self.dataset_type == "converted_coco":
            bboxes = tf.map_fn(lambda x: tf.strings.split(x, ','), line[1:])
            bboxes = tf.strings.to_number(bboxes, tf.float32)
        elif self.dataset_type == "yolo":
            height, width, _ = img.shape
            bboxes = tf.map_fn(lambda x: tf.strings.split(x, ','), line[1:])
            bboxes = tf.strings.to_number(bboxes, tf.float32)
            bboxes = bboxes * tf.constant([width, height, width, height, 1], dtype=tf.float32)
            bboxes = bboxes.astype(np.float32)
        # bboxes = bboxes * scale

        return img, bboxes
    
    @tf.function
    def random_crop(self, image, bboxes):
        """
        Parameter
        ---------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class
        Return
        ------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class
        """
        if tf.random.uniform((1,), 0, 1)[0] < 0.5:
            h =tf.shape(image)[0]
            w = tf.shape(image)[1]
            max_bbox = tf.concat(
                [
                    tf.reduce_min(bboxes[:, 0:2], axis=0),
                    tf.reduce_max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans =  tf.cast(w, dtype=tf.float32) - max_bbox[2]
            max_d_trans =  tf.cast(h, dtype=tf.float32) - max_bbox[3]

            crop_xmin = tf.maximum(
                0, tf.cast(max_bbox[0] - tf.random.uniform((1,), 0, max_l_trans)[0], tf.int32)
            )
            crop_ymin = tf.maximum(
                0, tf.cast((max_bbox[1] - tf.random.uniform((1,), 0, max_u_trans))[0], tf.int32)
            )
            crop_xmax = tf.maximum(
                w, tf.cast((max_bbox[2] + tf.random.uniform((1,), 0, max_r_trans))[0], tf.int32)
            )
            crop_ymax = tf.maximum(
                h, tf.cast((max_bbox[3] + tf.random.uniform((1,), 0, max_d_trans))[0], tf.int32)
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            bboxes = bboxes - tf.cast(tf.stack([crop_xmin,crop_ymin,crop_xmin,crop_ymin,0]), dtype=tf.float32)
        # tf.debugging.Assert(tf.reduce_all(bboxes >= 0.0), ['some of coordinate are negative!!'])
        return image, bboxes

    @tf.function
    def random_translate(self, image, bboxes):
        """
        Parameter
        ---------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class

        Return
        ------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class
        """
        if tf.random.uniform((1,), 0, 1)[0] < 0.5:
            h = tf.shape(image)[0]
            w = tf.shape(image)[1]
            max_bbox = tf.concat(
                [
                    tf.reduce_min(bboxes[:, 0:2], axis=0),
                    tf.reduce_max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = tf.cast(w, dtype=tf.float32) - max_bbox[2]
            max_d_trans = tf.cast(h, dtype=tf.float32) - max_bbox[3]

            tx = tf.random.uniform((1,),-(max_l_trans - 1), (max_r_trans - 1))[0]
            ty = tf.random.uniform((1,),-(max_u_trans - 1), (max_d_trans - 1))[0]

            image = tfa.image.translate(image, [tx,ty])
            bboxes = bboxes + tf.stack([tx, ty, tx, ty, 0])
        # tf.debugging.Assert(tf.reduce_all(bboxes >= 0.0), ['some of coordinate are negative!!'])
        return image, bboxes

    @tf.function
    def random_horizontal_flip(self, image, bboxes):
        """
        Parameter
        ---------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class

        Return
        ------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class
        """
        if tf.random.uniform((1,), 0, 1)[0] < 0.5:
            img_shape = tf.shape(image)
            w = tf.cast(img_shape[1], dtype=tf.float32)
            image = image[:, ::-1, :]
            bboxes = tf.concat([
                w-bboxes[:,2:3],
                  bboxes[:,1:2],
                w-bboxes[:,0:1],
                  bboxes[:,3:4],
                  bboxes[:,4:5]
            ], axis=-1)
        # tf.debugging.Assert(tf.reduce_all(bboxes >= 0.0), [['some of coordinate are negative!!']])
        return image, bboxes

    @tf.function
    def pad_image(self, img, bboxes):
        """
        Padding image to same size\\

        Parameter
        ---------
        img: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class
        
        Return
        ------
        img: uint8(h,w,3)\\
        bboxes: float32(num_of_bbox, 5)\\
            x1y1x2y2 class
        """
        train_input_size = self.train_input_sizes
        iw, ih = train_input_size, train_input_size
        origin_shape=tf.shape(img)
        h=origin_shape[0]
        w=origin_shape[1]
        scale = tf.cast(tf.minimum(train_input_size/w, train_input_size/h), dtype=tf.float32)
        nw, nh  = tf.math.round(scale * tf.cast(w, tf.float32)), tf.math.round(scale * tf.cast(h, tf.float32))
        if nw != train_input_size and nh != train_input_size:
            if nw > nh:
                nw = tf.cast(train_input_size, tf.float32)
            elif nw < nh:
                nh = tf.cast(train_input_size, tf.float32)
            else:
                nw = tf.cast(train_input_size, tf.float32)
                nh = tf.cast(train_input_size, tf.float32)
        nw = tf.cast(nw, tf.int32)
        nh = tf.cast(nh, tf.int32)
        ###########
        img = tf.cast(tf.image.resize(img, [nh, nw]), dtype=tf.uint8)

        # pad image
        # pw = tf.cast(tf.math.round((train_input_size - nw) / 2), dtype=tf.int32)
        # ph = tf.cast(tf.math.round((train_input_size - nh) / 2), dtype=tf.int32)
        pw =(train_input_size - nw) // 2
        ph =(train_input_size - nh) // 2
        if train_input_size - nh != 0:
            # padding top and bottom
            remain_h=ih-nh-ph
            img = tf.concat([tf.ones((ph,iw,3),tf.uint8)*128, img, tf.ones((remain_h,iw,3),tf.uint8)*128], axis=0)
        elif train_input_size - nw != 0:
            # padding left and right
            remain_w=iw-nw-pw
            img = tf.concat([tf.ones((ih,pw,3),tf.uint8)*128, img, tf.ones((ih,remain_w,3),tf.uint8)*128], axis=1)
        # Discard too small bounding box in origin image size
        # tf.debugging.Assert(tf.reduce_all(bboxes >= 0.0), ['some of coordinate are negative!!'])
        area = (bboxes[:,0] - bboxes[:,2]) * (bboxes[:,1] - bboxes[:,3])
        mask = area > self.filter_area
        bboxes = bboxes[mask]
        # Turn integer coordinate into padded image integer coordinate 
        bboxes = bboxes * scale
        bboxes = bboxes + tf.cast(tf.stack([pw,ph,pw,ph,0], axis=0), dtype=tf.float32)
        # Pad num of box to fix size
        # tf.debugging.Assert(tf.reduce_all(bboxes >= 0.0), ['some of coordinate are negative!!'])
        return img, bboxes

    @tf.function
    def pad_bbox(self, image, bboxes):
        """
        Parameter
        ---------
        image: uint8(h,w,3)\\
        bboxes: float32(num_of_box, 5)\\

        Return
        ------
        image: uint8(h,w,3)\\
        bboxes: float32(num_max_box, 5)
        """
        num_boxes=tf.shape(bboxes)[0]
        bboxes=tf.concat([bboxes, tf.zeros((self.max_bbox_per_scale-num_boxes,5), dtype=bboxes.dtype)], axis=0)
        return image, bboxes

    def do_augmentation(self, images, bboxes):
        """
        Parameter
        ---------
        images: uint8(b,h,w,3)
        bboxes: float32(b,num_of_max_bboxes, 5)\\
            coordinate in x1y1x2y2 integer coordinate

        Return
        ------
        aug_images: uint8(b,h,w,3)
        aug_bboxes: float32(b,num_of_max_box,4) x1y1x2y2
        """
        b, h, w = images.shape[:3]
        num_max_box= bboxes.shape[1]
        non_box_mask = np.all(bboxes[...,:4]==0.0, axis=-1, keepdims=True).astype(np.float32)

        division_rate=None
        #############################################################################################
        batch_bboxes_in_image = [
            ia.augmentables.bbs.BoundingBoxesOnImage.from_xyxy_array(bboxes[k, :, :4], (h,w,3)) 
            for k in range(b)
        ]
        # if division_rate==None:
        # aug_images <class 'numpy.ndarray'>
        # _aug_bboxes <class 'list'>
        aug_images, _aug_bboxes=self.aug_env(images=images, bounding_boxes=batch_bboxes_in_image)
        aug_bboxes=[xyxy.remove_out_of_image_fraction_(0.5).to_xyxy_array() for xyxy in _aug_bboxes]
        aug_bboxes=[np.concatenate([box, np.zeros((num_max_box-len(box), 4), np.float32)], axis=0) if len(box) != num_max_box else box  for box in aug_bboxes  ]
        aug_bboxes=np.array(aug_bboxes)
        # else:
        #     dr=division_rate
        #     batches = [
        #         ia.Batch(
        #             images=images[i*dr:i*dr+dr], 
        #             bounding_boxes=batch_bboxes_in_image[i*dr:i*dr+dr]
        #         )  for i in range(b//dr)
        #     ]
        #     aug_batch=list(self.aug_env.augment_batches(batches, background=True))
        #     aug_images = np.concatenate([aug_batch[i].images_aug for i in range(b//dr)], axis=0)
        #     _aug_batch_bboxes=[[aug_batch[i].bounding_boxes_aug[j].to_xyxy_array() for j in range(dr)] for i in range(b//dr)]
        #     _aug_batch_bboxes=[item for sublist in _aug_batch_bboxes for item in sublist]
        #     _aug_batch_bboxes=[np.concatenate([box, np.zeros((num_max_box-len(box), 4), np.float32)], axis=0) if len(box) != num_max_box else box  for box in _aug_batch_bboxes  ]
        #     aug_bboxes=np.array(_aug_batch_bboxes)

        #############################################################################################
        aug_bboxes=aug_bboxes * (1.0 - non_box_mask)
        aug_bboxes=np.concatenate([aug_bboxes, bboxes[...,4:]], axis=-1)
        return aug_images, aug_bboxes

    def generate_gth(self, batch_image, batch_label):
        """
        Parameter
        ---------
        batch_image: tf.uint8(b,h,w,c)\\
        batch_label: tf.float32(max_num_bbox, 5)\\

        Return
        ------
        batch_image: tf.float32(b,h,w,c)
        batch_label_bboxes[0]: tf.float32(b, fpn_size_b, fpn_size_b, num_of_anchor_per_scale, 5+num_of_class)
        batch_bboxes[0]: tf.float32(b, num_of_max_box, 4)
        batch_label_bboxes[1]: tf.float32(b, fpn_size_m, fpn_size_m, num_of_anchor_per_scale, 5+num_of_class)
        batch_bboxes[1]: tf.float32(b, num_of_max_box, 4)
        
        """
        batch_label = tf.cast(batch_label, dtype=tf.int32)
        batch_image = tf.cast(batch_image, dtype=tf.float32) / 255.
        
        self.train_input_size = cfg.TRAIN.INPUT_SIZE
        self.train_output_sizes = self.train_input_size // self.strides     # [input_size//stride1, input_size//stride2......]

        batch_label_bboxes = []         # [(batch, size, size, num_of_anchor, 5+num_of_class)......num_of_fpn]
        batch_bboxes = []               # [(batch, max_box_scale, 4)......num_of_fpn]
        for size in self.train_output_sizes:
            label_bbox = np.zeros((self.batch_size, size, size, self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_bboxes.append(label_bbox)
            batch_bbox = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_bboxes.append(batch_bbox)
        
        for num in range(self.batch_size):
            # bboxes=(num_of_boxes, 5)
            single_bboxes=batch_label[num]
            # label_bboxes=[(h,w,num_of_anchor_per_scale,6)...... num_of_fpn]
            # bboxes=[(num_of_max_bboxes, 4)...... num_of_fpn]
            target_output= tf.numpy_function(preprocess_true_boxes_jit, inp=[
                    single_bboxes,
                    len(self.train_output_sizes), 
                    self.train_output_sizes, 
                    self.anchor_per_scale,
                    self.num_classes,
                    self.max_bbox_per_scale,
                    self.strides,
                    self.anchors
                ], Tout=tf.float32)
            
            fpn_b=target_output[0]
            fpn_m=target_output[1]
            bboxes=target_output[2]
            label_bboxes=[fpn_b, fpn_m]

            for batch_bbox, bbox in zip(batch_bboxes, bboxes):
                batch_bbox[num,:,:] = bbox
            for batch_label_bbox, label_bbox in zip(batch_label_bboxes, label_bboxes):
                batch_label_bbox[num, :, :, :] = label_bbox

        return (
            batch_image,
            batch_label_bboxes[0], batch_bboxes[0],
            batch_label_bboxes[1], batch_bboxes[1]
        )        

    def __len__(self):
        return self.num_batchs



import numba as nb

@nb.njit
# (
    # nb.types.Tuple((
    #     nb.types.List(nb.float64[:,:,:,::1]), 
    #     nb.float64[:,:,::1]
    # ))
    # (nb.float32[:,:], nb.int32, nb.int32[:], nb.int32, nb.int32, nb.int32, nb.float32[:], nb.float32[:,:])
# )
def preprocess_true_boxes_jit(bboxes, num_fpn, train_output_sizes, anchor_per_scale, num_classes, max_bbox_per_scale, strides, anchors):
    """
    1. Generate dense anchor with ground truth
    2. Trun bbox x1y1x2y2 to xywh where xy is the center of bbox
    Parameter
    ----------
    bboxes: np.float32(num_of_bboxes, 5)\\
        the ground truth bbox including [x1y1x2y2 class], in integer coordinate
    num_fpn: int\\
        number of fpn
    train_output_sizes: [fpn1_size, fpn2_size]\\
        the size of fpn.(assume fpn is square)
    anchor_per_scale: int\\
         number of anchor in each fpn
    num_classes: int\\
        number of classes
    max_bbox_per_scale: int\\
        number of maximum bounding box in each fpn
    strides: [stride1, stride2]\\
        the stride for downsample image to fpn size
    anchors: 
        anchor size

    Return 
    ------
    label: list of np.float32(b,w,h,num_of_anchor_per_scale,5+num_of_class)
    bboxes_xywh: list of np.float32(b,max_num_of_bbox,4)\\
        xy is center of bbox, wh is width and height of bbox
    """
    num_fpn = num_fpn.item()
    anchor_per_scale = anchor_per_scale.item()
    num_classes = num_classes.item()
    max_bbox_per_scale = max_bbox_per_scale.item()


    label = [
        np.zeros(
            (
                train_output_sizes[i],
                train_output_sizes[i],
                anchor_per_scale,
                5 + num_classes,
            )
        )
        for i in range(num_fpn)
    ]
    bboxes_xywh = np.zeros((num_fpn, max_bbox_per_scale, 4))
    bbox_count = np.zeros((num_fpn,))
    for j in range(len(bboxes)):
        bbox = bboxes[j]
        bbox_coor = bbox[:4]
        if bbox_coor[2] == 0 and bbox_coor[3] == 0:
            break
        bbox_class_ind = np.int32(bbox[4])
        onehot = np.zeros((num_classes,), dtype=np.float32)
        onehot[bbox_class_ind] = np.float32(1.0)
        uniform_distribution = np.full( 
            num_classes, np.float32(1.0) / num_classes
        )
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
        bbox_xywh = np.concatenate(     # shape=(4,)
            (
                (bbox_coor[2:] + bbox_coor[:2]) * np.float32(0.5),
                bbox_coor[2:] - bbox_coor[:2],
            ),
            axis=-1,
        )

        bbox_xywh_scaled = (
            1.0 * np.expand_dims(bbox_xywh, axis=0) / np.expand_dims(strides, axis=-1)
        )
        

        iou = np.zeros((num_fpn, anchor_per_scale), dtype=np.float32)
        exist_positive = False
        for i in range(num_fpn):
            anchors_xywh = np.zeros((anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = (
                np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            )
            anchors_xywh[:, 2:4] = anchors[i]
            # iou_scale=(3,)
            iou_scale = utils.bbox_iou_jit(
                np.expand_dims(bbox_xywh_scaled[i], axis=0), anchors_xywh
            )
            iou[i] = iou_scale
            # iou_mask=(3,)
            iou_mask = iou_scale > 0.3
            if np.any(iou_mask):
                for k in range(anchor_per_scale):
                    if iou_mask[k]:
                        xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                            np.int32
                        )
                        label[i][yind, xind, k, :] = 0.0
                        label[i][yind, xind, k, 0:4] = bbox_xywh
                        label[i][yind, xind, k, 4:5] = 1.0
                        label[i][yind, xind, k, 5:] = smooth_onehot
                        bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                        bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                        bbox_count[i] += 1
                exist_positive = True
                
        if not exist_positive:
            iou_flatten = iou.reshape(-1)
            best_anchor_ind = np.argmax(iou_flatten, axis=-1)
            best_detect = int(best_anchor_ind / anchor_per_scale) # the best match fpn index
            best_anchor = int(best_anchor_ind % anchor_per_scale) # the best match anchor in certain fpn
            xind, yind = np.floor(
                bbox_xywh_scaled[best_detect, 0:2]
            ).astype(np.int32)
            label[best_detect][yind, xind, best_anchor, :] = 0.0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
            bbox_ind = int(
                bbox_count[best_detect] % max_bbox_per_scale
            )
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    #label_sbbox, label_mbbox, label_lbbox = label
    #sbboxes, mbboxes, lbboxes = bboxes_xywh
    return label, bboxes_xywh #label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes