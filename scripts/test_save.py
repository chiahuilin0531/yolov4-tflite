import sys
import cv2
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator
import os, shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train, filter_boxes
from core.dataset_tiny import Dataset, tfDataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import draw_bbox, freeze_all, unfreeze_all, read_class_names
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
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
    # if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
    #     print(f'{layer.name:30s} {True}')
    #     return layer
    # if 'lambda' in layer.name:
    #     print(f'{layer.name:30s} {True} lambda !!')
    #     return layer
    if 'tf_op' in layer.name or 'lambda' in layer.name or \
        'tf.' in layer.name or 'activation' in layer.name or \
            'multiply' in layer.name:
        print(f'{layer.name:30s} {True}')
        return layer
    print(f'{layer.name:30s} {False}')
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
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # trainset = Dataset(FLAGS, is_training=True, filter_area=123)
    # testset = Dataset(FLAGS, is_training=False, filter_area=123)
    #########################################################
    # use_imgaug augmentation would lead to unknown performance drop
    # this issue should be resolved in the future.
    # trainset = tfDataset(FLAGS, cfg, is_training=True, filter_area=123, use_imgaug=False).dataset_gen(repeat_times=int(FLAGS.repeat_times))
    # testset = tfDataset(FLAGS, cfg, is_training=False, filter_area=123, use_imgaug=False).dataset_gen()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'config'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'pic'), exist_ok=True)


    copytree('./core', os.path.join(FLAGS.save_dir, 'config'))
    shutil.copy2(sys.argv[0], os.path.join(FLAGS.save_dir, 'config', os.path.basename(sys.argv[0])))
    with open(os.path.join(FLAGS.save_dir, 'command.txt'), 'w') as f:
        f.writelines(' '.join(sys.argv))
    f.close()


    # logdir = os.path.join(FLAGS.save_dir, 'logs')
    # isfreeze = False
    # steps_per_epoch = len(trainset)
    # first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    # second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    # global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    # warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    # total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    ckpt_path = os.path.join(FLAGS.save_dir, 'ckpt', '0000.ckpt')
    save_model_path = os.path.join(FLAGS.save_dir, 'save_model_tflite')

    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    # Section1 Save Ckpt Weights
    #################################################################################################################################################
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS, cfg)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    classes_name = read_class_names(cfg.YOLO.CLASSES )

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)

    # Decoding YOLOv4 Output
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, {
        'raw_bbox_m': bbox_tensors[0],          # tensor size of feature map
        'bbox_m': bbox_tensors[1],
        'raw_bbox_l': bbox_tensors[2],
        'bbox_l': bbox_tensors[3],
    })

    if FLAGS.weights == None:
        print("Training from scratch ......................")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)
    
    #####################################################################################################
    if (FLAGS.qat):
        model = qa_train(model)
        print("Training in Quatization Aware ................. ")
    #####################################################################################################
    model.save_weights(ckpt_path)
    model.summary()
    del model

    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    # Section2 Save Save_Model Format
    #################################################################################################################################################
    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)


    framework='tflite'
    bbox_tensors = []
    prob_tensors = []
    if FLAGS.tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            elif i == 1:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors[::-1], axis=1)
    pred_prob = tf.concat(prob_tensors[::-1], axis=1)
    if framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    elif framework == 'tf':
        pred = (pred_bbox, pred_prob)

    else:
        raise NotImplementedError(f'No such framework {FLAGS.framework}')
    print('========================================== Load model ========================================')
    model = tf.keras.Model(input_layer, pred)
    print (FLAGS.model+"      ....................................... ")
    if (FLAGS.qat):
        model = qa_train(model)
        print('Quantization Model .....................................')

    print('========================================== Load weight ========================================')
    model.load_weights(ckpt_path) # './checkpoints/day_tw_qat/ckpt/0057.ckpt'
    np.random.seed(100)
    res = model(np.random.uniform(size=(1,608,608,3)))
    print(res[0], res[0].numpy().sum())
    print(res[1])

    model.save(save_model_path)

    exit()
    #################################################################################################################################################
    #################################################################################################################################################
    #################################################################################################################################################
    # Section3 Tflite Format
    #################################################################################################################################################
    
    fp32 = False
    int8 = True
    
    if fp32:
        converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']

        np.random.seed(90)
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = np.ones(input_shape, dtype=np.float32)*3.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        print('Successfully inference fp32 model')
        del converter
        del interpreter

    if int8:
        def representative_data_gen():
            for input_value in range(5):
                image_data = np.random.uniform(size=(FLAGS.input_size, FLAGS.input_size, 3))
                img_in = image_data[np.newaxis, ...].astype(np.float32)
                yield [img_in]
        converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']

        np.random.seed(90)
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = np.ones(input_shape, dtype=np.float32)*3.0

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
