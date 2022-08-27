import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, decode_train, filter_boxes
from core.iayolo import CNNPP, DIP_FilterGraph
import core.utils as utils
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_boolean('iayolo', False, 'use IAYOLO or not')
flags.DEFINE_integer('input_size', 608, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('dataset', 'data/dataset/gis_val_1.txt', 'yolov3 or yolov4')
flags.DEFINE_boolean('qat', False, 'For Qauntize Aware Training')
flags.DEFINE_string('config_name', 'core.config', 'configuration ')
tf.config.optimizer.set_jit(True)


def apply_quantization(layer):
    # if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):

    # if 'tf_op' in layer.name or 'lambda' in layer.name or\
    #    'tf.' in layer.name or 'activation' in layer.name or\
    #     'multiply' in layer.name:
    #   return layer
    if 'tf_op' in layer.name or 'lambda' in layer.name or \
        'tf.' in layer.name or isinstance(layer, tfa.layers.InstanceNormalization) or \
            'multiply' in layer.name:
        return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

    # if isinstance(layer , tf.keras.layers.Conv2D):
    #     return tfmot.quantization.keras.quantize_annotate_layer(layer)
    # return layer

def qa_train(model):
    # qa_train part
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    
    annotated_model = tf.keras.models.clone_model(model,
        clone_function=apply_quantization,
    )

    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    # quant_aware_model.summary()

    return quant_aware_model

def save_tf():
  import importlib
  cfg = importlib.import_module(FLAGS.config_name).cfg

  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS, cfg)

  input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
  if FLAGS.iayolo:
      resized_input = tf.image.resize(input_layer, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
      filter_parameters = CNNPP(resized_input)
      yolo_input, processed_list = DIP_FilterGraph(input_layer, filter_parameters)
  else:
      yolo_input = input_layer
  feature_maps = YOLO(yolo_input, NUM_CLASS, FLAGS.model, FLAGS.tiny, nl=cfg.YOLO.NORMALIZATION)
  bbox_tensors = []
  prob_tensors = []
  if FLAGS.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      elif i == 1:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors[::-1], axis=1)
  pred_prob = tf.concat(prob_tensors[::-1], axis=1)
  # pred_bbox = tf.concat(bbox_tensors, axis=1)
  # pred_prob = tf.concat(prob_tensors, axis=1)
  if FLAGS.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  elif FLAGS.framework == 'tf':
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  else:
    raise NotImplementedError(f'No such framework {FLAGS.framework}')
  ## ori code
  print('========================================== Load model ========================================')
  # model = tf.keras.Model(input_layer, pred)
  if FLAGS.framework == 'tf':
    model = tf.keras.Model(input_layer, [pred, yolo_input])
  elif FLAGS.framework == 'tflite':
    model = tf.keras.Model(input_layer, pred)
    
  ##################################################################################################### 
  # Decoding YOLOv4 Output
  # if FLAGS.tiny:
  #     bbox_tensors = []
  #     for i, fm in enumerate(feature_maps):
  #         if i == 0:
  #             bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
  #         else:
  #             bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
  #         bbox_tensors.append(fm)
  #         bbox_tensors.append(bbox_tensor)
  # else:
  #     bbox_tensors = []
  #     for i, fm in enumerate(feature_maps):
  #         if i == 0:
  #             bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
  #         elif i == 1:
  #             bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
  #         else:
  #             bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
  #         bbox_tensors.append(fm)
  #         bbox_tensors.append(bbox_tensor)
  # output_dict = {
  #     'raw_bbox_m': bbox_tensors[0],          # tensor size of feature map
  #     'bbox_m': bbox_tensors[1],
  #     'raw_bbox_l': bbox_tensors[2],
  #     'bbox_l': bbox_tensors[3],
  # }
  # if FLAGS.iayolo: output_dict.update({'dip_img': yolo_input})
  # model = tf.keras.Model(input_layer, output_dict)
  
  ############################################ Quantize structure #####################################
  print (FLAGS.model+"      ....................................... ")
  if (FLAGS.qat):
    model = qa_train(model)
    print('Quantization Model .....................................')
  #####################################################################################################

  print('========================================== Load weight ========================================')
  model.load_weights(FLAGS.weights)
  print('========================================== model summary ========================================')
  model.summary()
  print('========================================== save model ========================================')
  model.save(FLAGS.output)
  # if FLAGS.iayolo and FLAGS.framework != 'tflite':
  #   for name in ['ex_conv0_conv2d', 'ex_conv1_conv2d', 'ex_conv2_conv2d', 'ex_conv3_conv2d', 'ex_conv4_conv2d', 'dense', 'dense_1']:
  #     print('*'*40)
  #     ly=model.get_layer(name)
  #     for i in range(len(ly.get_weights())):
  #       w=ly.get_weights()[i]
  #       print(w.shape, tf.reduce_sum(w))
  
  
    # test=input("wait for u")
  # for i in range(10):
  #   print(model.get_layer(index=i).get_weights())

  # converter = tf.lite.TFLiteConverter.from_keras_model(model)

  # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.allow_custom_ops = True
  # converter.representative_dataset = representative_data_gen

  # tflite_model = converter.convert()
  # open("./checkpoints/yolov4-tiny-gis-320-int8.tflite", 'wb').write(tflite_model)

  # flops = get_flops(model, batch_size=1)
  # print(f"FLOPS: {flops / 10 ** 9:.03} G")
  import cv2
  import numpy as np
  "./checkpoints/day_tw_qat_v2/save_model_0056/"
  img = cv2.imread('./datasets/data_selection/images/image_000014.jpg')
  img = utils.image_preprocess(img, (608,608))
  img = np.expand_dims(img, axis=0)
  bef_res = model(img, training=False)
  infer = tf.keras.models.load_model(FLAGS.output)
  aft_res = infer(img)

  print('before convert result')
  print(bef_res)
  print('after convert result')
  print(aft_res)

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
