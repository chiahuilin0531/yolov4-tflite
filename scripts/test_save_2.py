import sys
import cv2
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator
import os, shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.config import cfg
import numpy as np
from core import utils
from tqdm import tqdm
# import tensorflow_model_optimization as tfmot
import time

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('qat', True, 'train w/ or w/o quatize aware')
flags.DEFINE_string('save_dir', 'checkpoints/test_save', 'save model dir')
flags.DEFINE_float('repeat_times', 1.0, 'repeat of dataset')
flags.DEFINE_integer('input_size', 608, 'repeat of dataset')

# tf.config.optimizer.set_jit(True)
import tensorflow.keras as keras
# import keras
def apply_quantization(layer):
    # print(layer.name, isinstance(layer, tf.python.keras.layers.core.TFOpLambda))
    # if isinstance(layer, tf.python.keras.layers.core.TFOpLambda):
    #     return layer
    # else:
    #     return tfmot.quantization.keras.quantize_annotate_layer(layer)
    
    # if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
    if 'tf_op' in layer.name or 'lambda' in layer.name or 'tf.' in layer.name or 'activation' in layer.name:
        print(layer.name)
        return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

def qa_train(model):
    # qa_train part
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    
    annotated_model = tf.keras.models.clone_model(model,
        clone_function=apply_quantization,
    )

    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return quant_aware_model

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.rmtree(d, ignore_errors=True)
            shutil.copy2(s, d)

def main(_argv):
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')


    # os.makedirs(FLAGS.save_dir, exist_ok=True)
    # os.makedirs(os.path.join(FLAGS.save_dir, 'config'), exist_ok=True)
    # os.makedirs(os.path.join(FLAGS.save_dir, 'ckpt'), exist_ok=True)
    # os.makedirs(os.path.join(FLAGS.save_dir, 'pic'), exist_ok=True)


    # copytree('./core', os.path.join(FLAGS.save_dir, 'config'))
    # shutil.copy2(sys.argv[0], os.path.join(FLAGS.save_dir, 'config', os.path.basename(sys.argv[0])))
    # with open(os.path.join(FLAGS.save_dir, 'command.txt'), 'w') as f:
    #     f.writelines(' '.join(sys.argv))
    # f.close()


    ckpt_path = os.path.join(FLAGS.save_dir, 'ckpt', '0000.ckpt')
    save_model_path = os.path.join(FLAGS.save_dir, 'save_model_tflite')

    # #################################################################################################################################################
    # #################################################################################################################################################
    # #################################################################################################################################################
    # # Section1 Save Ckpt Weights
    # #################################################################################################################################################
    # input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS, cfg)
    # IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    # freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    # feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)

    # # Decoding YOLOv4 Output
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
    # model = tf.keras.Model(input_layer, {
    #     'raw_bbox_m': bbox_tensors[0],          # tensor size of feature map
    #     'bbox_m': bbox_tensors[1],
    #     'raw_bbox_l': bbox_tensors[2],
    #     'bbox_l': bbox_tensors[3],
    # })

    # if FLAGS.weights == None:
    #     print("Training from scratch ......................")
    # else:
    #     if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
    #         utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    #     else:
    #         model.load_weights(FLAGS.weights)
    #     print('Restoring weights from: %s ... ' % FLAGS.weights)
    
    # #####################################################################################################
    # if (FLAGS.qat):
    #     model = qa_train(model)
    #     print("Training in Quatization Aware ................. ")
    # #####################################################################################################
    # model.save_weights(ckpt_path)
    # del model

    # #################################################################################################################################################
    # #################################################################################################################################################
    # #################################################################################################################################################
    # # Section2 Save Save_Model Format
    # #################################################################################################################################################

    # framework='tflite'
    # input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    # feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    # bbox_tensors = []
    # prob_tensors = []
    # if FLAGS.tiny:
    #     for i, fm in enumerate(feature_maps):
    #         if i == 0:
    #             output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
    #         else:
    #             output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
    #         bbox_tensors.append(output_tensors[0])
    #         prob_tensors.append(output_tensors[1])
    # else:
    #     for i, fm in enumerate(feature_maps):
    #         if i == 0:
    #             output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
    #         elif i == 1:
    #             output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
    #         else:
    #             output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
    #         bbox_tensors.append(output_tensors[0])
    #         prob_tensors.append(output_tensors[1])
    # pred_bbox = tf.concat(bbox_tensors[::-1], axis=1)
    # pred_prob = tf.concat(prob_tensors[::-1], axis=1)
    # if framework == 'tflite':
    #     pred = (pred_bbox, pred_prob)
    # elif framework == 'tf':
    #     boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    #     pred = tf.concat([boxes, pred_conf], axis=-1)
    # else:
    #     raise NotImplementedError(f'No such framework {FLAGS.framework}')
    # print('========================================== Load model ========================================')
    # model = tf.keras.Model(input_layer, pred)
    # print (FLAGS.model+"      ....................................... ")
    # if (FLAGS.qat):
    #     model = qa_train(model)
    #     print('Quantization Model .....................................')

    # print('========================================== Load weight ========================================')
    # model.load_weights(ckpt_path)
    # print('========================================== save model ========================================')
    # model.save(save_model_path)

    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    # Section3 Tflite Format
    #################################################################################################################################################
    
    fp32 = True
    int8 = True
    
    if fp32:
        converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
        tflite_model = converter.convert()
        open(os.path.join(FLAGS.save_dir, 'float32.tflite'), 'wb').write(tflite_model)

        interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=8)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        idx=299
        print(f'FINAL _get_tensor_details({idx})', interpreter._get_tensor_details(idx))


        np.random.seed(90)
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = np.ones(input_shape, dtype=np.float32)*0.5

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        for i in range(len(output_details)):
            print(f'{i} output tensor shape ',interpreter.get_tensor(output_details[i]['index']).shape)
        print('Successfully inference fp32 model')
        del converter
        del interpreter

    if int8:
        def representative_data_gen():
            fimage = open('../yolov4-tflite-new/datasets/data_selection_mix/anno/val_3cls.txt').readlines()
            fimage = [line.split()[0] for line in fimage]
            np.random.seed(0)
            # np.random.seed(49)
            # np.random.seed(100)
            np.random.shuffle(fimage)
            for input_value in range(5):
                if os.path.exists(fimage[input_value]):
                    original_image=cv2.imread(fimage[input_value])
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    # Processing V1
                    image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
                    #####################################################################################################
                    # Processing V2
                    # image_data = cv2.resize(np.copy(original_image), (FLAGS.input_size, FLAGS.input_size))
                    # image_data = image_data / 255.0
                    #####################################################################################################
                    img_in = image_data[np.newaxis, ...].astype(np.float32)
                    print(f"{input_value} {img_in.shape} calibration image {fimage[input_value]}")
                    yield [img_in]
                else:
                    print('not found', fimage[input_value])
                    continue
        converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen
        tflite_model = converter.convert()
        open(os.path.join(FLAGS.save_dir, 'int8.tflite'), 'wb').write(tflite_model)

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']

        np.random.seed(90)
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = np.ones(input_shape, dtype=np.float32)*0.5

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        print('Successfully inference int8 model')
        del converter
        del interpreter

    # 

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
