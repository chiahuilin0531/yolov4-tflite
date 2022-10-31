from audioop import reverse
import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from tqdm import tqdm
FLAGS = edict()

FLAGS.weights ='./checkpoints/0719_test1/save_model_final_tflite/'
FLAGS.export_dir = 'tflite_exp_v2'
FLAGS.input_size =608
FLAGS.dataset ="datasets/data_selection_mix/anno/val_3cls.txt"
quantilize_with_relu = True



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

def add_statistic(filename):
    layer_stats = pd.read_csv(filename)
    print(layer_stats.head())
    layer_stats['range'] = 255.0 * layer_stats['scale']
    layer_stats['rmse/scale'] = layer_stats.apply(
        lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
    return layer_stats


if not os.path.exists(FLAGS.export_dir):
    os.makedirs(FLAGS.export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_data_gen)
debugger.run()

RESULTS_FILE = './debugger_results.csv'
RESULTS_FILE_V2 = './debugger_results_V2.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
layer_stats = add_statistic(RESULTS_FILE)
layer_stats.to_csv(RESULTS_FILE_V2)
relu_ops = [item for item in layer_stats['tensor_name'] if 'relu' in item]
if quantilize_with_relu:
    origin_int8_model = debugger.get_nondebug_quantized_model()
    num_of_bytes = open(os.path.join(FLAGS.export_dir,'origin_int8_model.tflite'), 'wb').write(origin_int8_model)
    print(f'[Info] origin model {num_of_bytes} bytes')
else:
    debug_options = tf.lite.experimental.QuantizationDebugOptions(
        denylisted_nodes=relu_ops
    )
    debugger = tf.lite.experimental.QuantizationDebugger(
        debug_options=debug_options,
        converter=converter, 
        debug_dataset=representative_data_gen)
    debugger.run()
    
    origin_int8_model = debugger.get_nondebug_quantized_model()
    num_of_bytes = open(os.path.join(FLAGS.export_dir,'origin_int8_model_no_relu.tflite'), 'wb').write(origin_int8_model)
    print(f'[Info] num of relu operation {len(relu_ops)}')
    print(f'[Info] origin model {num_of_bytes} bytes')

# import subprocess
# 1,  30 =>
# 31, 40 =>
end=len(list(layer_stats['tensor_name']))
for last_k in range(33,34):
    st = end - last_k
    suspected_layers = list(layer_stats['tensor_name'])[st:end]
    if not quantilize_with_relu:
        suspected_layers += relu_ops
        suspected_layers = list(set(suspected_layers))

    debug_options = tf.lite.experimental.QuantizationDebugOptions(
        denylisted_nodes=suspected_layers)
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter,
        debug_dataset=representative_data_gen,
        debug_options=debug_options)

    if quantilize_with_relu:
        filename = os.path.join(FLAGS.export_dir,f'selective_int8_model_{len(suspected_layers)}_layer_st{st}_end{end}.tflite')
    else:
        filename = os.path.join(FLAGS.export_dir,f'selective_int8_model_{len(suspected_layers)}_layer_st{st}_end{end}_no_relu.tflite')

    selective_quantized_model_dbg = debugger.get_nondebug_quantized_model()
    with open(filename, 'wb') as f:
        num_of_bytes = f.write(selective_quantized_model_dbg)
    f.close()
    print(f'[Info] selective model {num_of_bytes} bytes. {len(suspected_layers)} layers')
    # subprocess.run(['python', 'evaluate_map_v3.py', '--weights', filename, '--framework', 'tflite'])