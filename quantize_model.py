from audioop import reverse
import os
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from tqdm import tqdm

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_size', 608, 'path to output')
flags.DEFINE_string('dataset', "datasets/data_selection_mix/anno/val_3cls.txt", 'path to dataset')

def representative_data_gen():
    len_img=10
    fimage = open(FLAGS.dataset).readlines()
    fimage = [line.split()[0] for line in fimage]
    np.random.seed(0)
    np.random.shuffle(fimage)
    with tqdm(total=len_img, ncols=100) as pbar:
        for input_value in range(len_img):
            if os.path.exists(fimage[input_value]):
                original_image=cv2.imread(fimage[input_value])
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Processing V1
                # image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
                #####################################################################################################
                # Processing V2
                image_data = cv2.resize(np.copy(original_image), (FLAGS.input_size, FLAGS.input_size))
                image_data = image_data / 255.0
                #####################################################################################################
                img_in = image_data[np.newaxis, ...].astype(np.float32)
                pbar.set_postfix({
                    'image': fimage[input_value]
                })
                pbar.update(1)
                yield [img_in]
            else:
                pbar.set_postfix({
                    'image': ''
                })
                pbar.update(1)

def main(_argv):
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    ################################
    # Construct Int8 Intepreter
    ################################
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.allow_custom_ops = True
    converter_int8.experimental_new_converter = True
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_data_gen
    
    ################################
    # Construct Int8 Debugger
    ################################
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter_int8,
        debug_dataset=representative_data_gen)
    debugger.run()

    # RESULTS_FILE = './debugger_results.csv'
    # with open(RESULTS_FILE, 'w') as f:
    #     debugger.layer_statistics_dump(f)
    # layer_stats = add_statistic(RESULTS_FILE)
    # lst = list(layer_stats['tensor_name'])
    # end=len(list(layer_stats['tensor_name']))
    
    ################################
    # Selected non-quantilize layer
    ################################
    last_k = 33
    op_names = [debugger._get_operand_name_and_index(name)[0] for name, metrics in debugger.layer_statistics.items()]
    
    end = len(op_names)
    st = end - last_k
    suspected_layers = op_names[-last_k:]

    ################################
    # Quantilize Partial Model
    ################################
    debug_options = tf.lite.experimental.QuantizationDebugOptions(
        denylisted_nodes=suspected_layers)
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter_int8,
        debug_dataset=representative_data_gen,
        debug_options=debug_options)
    
    ################################
    # Save Tflite Int8 Model
    ################################
    filename = os.path.join(FLAGS.output,f'selective_int8_model_{len(suspected_layers)}_layer_st{st}_end{end}.tflite')
    model = debugger.get_nondebug_quantized_model()
    with open(filename, 'wb') as f:
        num_of_bytes = f.write(model)
    f.close()
    print(f'[Info] selective model {filename} {num_of_bytes} bytes. {len(suspected_layers)} layers')
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass