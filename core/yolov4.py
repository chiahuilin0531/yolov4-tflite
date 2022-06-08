#! /usr/bin/env python
# coding=utf-8

import enum
import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False, dc_head_type=0):
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS, dc_head_type)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == 'yolov4':
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)

def YOLOv3(input_layer, NUM_CLASS, use_dc_head):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer, NUM_CLASS, use_dc_head):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_tiny(input_layer, NUM_CLASS, dc_head_type):
    route_1, conv = backbone.cspdarknet53_tiny(input_layer)
    if dc_head_type == 1:
        conv_mdc, conv_ldc = DomainClassifier([route_1, conv])
    elif dc_head_type == 2:
        conv_mdc, conv_ldc = [route_1, conv]

    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    if dc_head_type == 0:
        return [conv_mbbox, conv_lbbox]
    else:
        return [conv_mbbox, conv_lbbox, conv_mdc, conv_ldc]

def YOLOv3_tiny(input_layer, NUM_CLASS, use_dc_head):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1,1,1], FRAMEWORK='tf'):
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tflite':
        return decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'tf':
        return decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        raise NotImplementedError(f'No such framewrok {FRAMEWORK}')

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_tflite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0,\
    conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1,\
    conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = tf.split(conv_output, (2, 2, 1+NUM_CLASS, 2, 2, 1+NUM_CLASS,
                                                                                2, 2, 1+NUM_CLASS), axis=-1)

    conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
    for idx, score in enumerate(conv_raw_score):
        score = tf.sigmoid(score)
        score = score[:, :, :, 0:1] * score[:, :, :, 1:]
        conv_raw_score[idx] = tf.reshape(score, (1, -1, NUM_CLASS))
    pred_prob = tf.concat(conv_raw_score, axis=1)

    conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
    for idx, dwdh in enumerate(conv_raw_dwdh):
        dwdh = tf.exp(dwdh) * ANCHORS[i][idx]
        conv_raw_dwdh[idx] = tf.reshape(dwdh, (1, -1, 2))
    pred_wh = tf.concat(conv_raw_dwdh, axis=1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.stack(xy_grid, axis=-1)  # [gx, gy, 2]
    xy_grid = tf.expand_dims(xy_grid, axis=0)
    xy_grid = tf.cast(xy_grid, tf.float32)

    conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
    for idx, dxdy in enumerate(conv_raw_dxdy):
        dxdy = ((tf.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
        conv_raw_dxdy[idx] = tf.reshape(dxdy, (1, -1, 2))
    pred_xy = tf.concat(conv_raw_dxdy, axis=1)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    # x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=0), [output_size, 1])
    # y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.float32), axis=1), [1, output_size])
    # xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    # pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
    #           STRIDES[i]
    pred_xy = (tf.reshape(tf.sigmoid(conv_raw_dxdy), (-1, 2)) * XYSCALE[i] - 0.5 * (XYSCALE[i] - 1) + tf.reshape(xy_grid, (-1, 2))) * STRIDES[i]
    pred_xy = tf.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob

    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob
    # return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    """
    Parameters
    ----------
    box_xywh: (None, None, 4)
    scores: (None, None, num_of_cls)
    """
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0, IOU_LOSS=utils.bbox_giou):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    input_size = tf.cast(input_size, tf.float32)
    
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

    giou = tf.expand_dims(IOU_LOSS(pred_xywh, label_xywh), axis=-1)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    alpha = 0.5
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss

def compute_da_loss(source_da_result, target_da_result, source_mask=None, target_mask=None):
    da_source_loss=0
    for i, source_da_res in enumerate(source_da_result):
        source_label=tf.zeros((tf.shape(source_da_res)))
        tmp=tf.nn.sigmoid_cross_entropy_with_logits(labels=source_label, logits=source_da_res)

        if isinstance(source_mask, list):
            tmp = tmp * source_mask[i]
            da_source_loss += (tf.reduce_sum(tmp, axis=[1,2,3]) / (tf.reduce_sum(source_mask[i], axis=[1,2,3]) + 1))
        elif source_mask is not None:
            resize_mask = tf.image.resize(source_mask, tf.shape(source_label)[1:3])
            tmp = tmp * resize_mask
            da_source_loss += (tf.reduce_sum(tmp, axis=[1,2,3]) / (tf.reduce_sum(resize_mask, axis=[1,2,3])))
        else:
            da_source_loss += tf.reduce_mean(tmp, axis=[1,2,3])
    
    da_target_loss=0
    for j, target_da_res in enumerate(target_da_result):
        target_label=tf.ones((tf.shape(target_da_res)))
        tmp=tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label, logits=target_da_res)

        if isinstance(target_mask, list):
            tmp = tmp * target_mask[j]
            da_target_loss += (tf.reduce_sum(tmp, axis=[1,2,3]) / (tf.reduce_sum(target_mask[j], axis=[1,2,3]) + 1))
        elif target_mask is not None: 
            resize_mask = tf.image.resize(target_mask, tf.shape(target_label)[1:3])
            tmp = tmp * resize_mask
            da_target_loss += (tf.reduce_sum(tmp, axis=[1,2,3]) / (tf.reduce_sum(resize_mask, axis=[1,2,3])))
        else:
            da_target_loss += tf.reduce_mean(tmp, axis=[1,2,3]) 

    da_source_loss = tf.reduce_mean(da_source_loss, axis=0)
    da_target_loss = tf.reduce_mean(da_target_loss, axis=0)

    return {
        'da_source_loss': da_source_loss,
        'da_target_loss': da_target_loss
    }

def compute_da_loss_instance(source_da_result, target_da_result, source_mask=None, target_mask=None):
    da_source_loss=0
    for i, source_da_res in enumerate(source_da_result):
        source_label=tf.zeros((tf.shape(source_da_res)))
        tmp=tf.nn.sigmoid_cross_entropy_with_logits(labels=source_label, logits=source_da_res)

        if isinstance(source_mask, list):
            tmp = tmp * source_mask[i]
            da_source_loss += tf.math.divide_no_nan(tf.reduce_sum(tmp), tf.reduce_sum(source_mask[i]))
    
    da_target_loss=0
    for j, target_da_res in enumerate(target_da_result):
        target_label=tf.ones((tf.shape(target_da_res)))
        tmp=tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label, logits=target_da_res)

        if isinstance(target_mask, list):
            tmp = tmp * target_mask[j]
            da_target_loss += tf.math.divide_no_nan(tf.reduce_sum(tmp), tf.reduce_sum(target_mask[j]))

    # da_source_loss = tf.reduce_mean(da_source_loss, axis=0)
    # da_target_loss = tf.reduce_mean(da_target_loss, axis=0)

    return {
        'da_source_loss': da_source_loss,
        'da_target_loss': da_target_loss
    }

def DomainClassifier(input_tensors):
    # return list of processing symbolic tensor
    reuslt_tensor=[]
    for i,input_tensor in enumerate(input_tensors):
        channel = input_tensor.shape[-1]
        x = common.GradientReversal(0.1)(input_tensor)
        x = common.convolutional(x, (3, 3, channel, channel // 2), prefix=f'domain_adverserial_{i}_0')
        x = common.convolutional(x, (1, 1, channel // 2, 1), activate=False, bn=False, prefix=f'domain_adverserial_{i}_1')
        reuslt_tensor.append(x)
    return reuslt_tensor
