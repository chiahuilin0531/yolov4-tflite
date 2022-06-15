import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
import core.utils as utils
import os

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, float16, float32)')
flags.DEFINE_string('dataset', "/Volumes/Elements/data/coco_dataset/coco/5k.txt", 'path to dataset')

def representative_data_gen():
  fimage = open(FLAGS.dataset).readlines()
  fimage = [line.split()[0] for line in fimage]
  np.random.shuffle(fimage)
  for input_value in range(10):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      print(f"{input_value} {img_in.shape} calibration image {fimage[input_value]}")
      yield [img_in]
    else:
      print('not found', fimage[input_value])
      continue

def save_tflite():
  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)

  if FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
  elif FLAGS.quantize_mode == 'int8':
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8, 
      tf.lite.OpsSet.TFLITE_BUILTINS, 
      tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  logging.info("=================== model saved to: {}  ==================".format(FLAGS.output))

def demo():
  logging.info(f'demo function. load model from {FLAGS.output}')
  interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  print('input_details')
  for i in range(len(input_details)): print('\t',input_details[i])
  output_details = interpreter.get_output_details()
  print('output_details')
  for i in range(len(output_details)): print('\t',output_details[i])

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  print('input_data', input_data.shape)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  # output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
  print("output_details[0]['index']", output_details[0]['index'])
  output_data = [interpreter.get_tensor(output_details[0]['index'])]

  print(output_data)

def main(_argv):
  save_tflite()
  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


