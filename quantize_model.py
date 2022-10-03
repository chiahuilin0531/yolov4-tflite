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
import  core.utils as utils

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/tflite/', 'output directory')
flags.DEFINE_integer('input_size', 608, 'path to output')
flags.DEFINE_string('dataset', "datasets/data_selection_mix/anno/val_3cls.txt", 'path to dataset')
flags.DEFINE_boolean('stats', False, 'show the quantization error statistics')


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
                image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
                #####################################################################################################
                # Processing V2
                # image_data = cv2.resize(np.copy(original_image), (FLAGS.input_size, FLAGS.input_size))
                # image_data = image_data / 255.0
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


def main(_argv):
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    ################################
    # Construct Int8 Intepreter
    ################################
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)
    converter_int8.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        # tf.lite.OpsSet.SELECT_TF_OPS, 
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter_int8.allow_custom_ops = False
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

    if FLAGS.stats:
        STATS_FILE = 'debugger_results.csv'
        STATS_PATH = os.path.join(FLAGS.output, STATS_FILE)
        with open(STATS_PATH, 'w') as f:
            debugger.layer_statistics_dump(f)
        layer_stats = add_statistic(STATS_PATH)
        lst = list(layer_stats['tensor_name'])
        end=len(list(layer_stats['tensor_name']))
        layer_stats.to_csv(STATS_PATH+'_v2.csv')
    
    ################################
    # Selected non-quantilize layer
    ################################
    last_k = 33
    op_names = [debugger._get_operand_name_and_index(name)[0] for name, metrics in debugger.layer_statistics.items()]
    
    end = len(op_names)
    st = end - last_k
    suspected_layers = op_names[-last_k:]
    
    ############################################
    # If Some Layer Degrade the Performance
    # Then We May Need To Add Extra Layer 
    # To Non-Quantization List
    ############################################
    # suspected_layers += op_names[43:44]
    # fail case: suspected_layers += op_names[13:25]
    # success case: only += op_names[13:113], only += op_names[13:57]
    # suspected_layers += op_names[13:113]
    # suspected_layers += op_names[:11]
    
    
    print('op_names', len(op_names))
    print('suspected_layers', len(suspected_layers))
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
    filename = os.path.join(FLAGS.output,f'int8_model_{len(suspected_layers)}_layer_st{st}_end{end}.tflite')
    model = debugger.get_nondebug_quantized_model()
    with open(filename, 'wb') as f:
        num_of_bytes = f.write(model)
    f.close()
    
    with open(os.path.join(FLAGS.output, 'readme.txt'), 'w') as f:
        for layer_name in suspected_layers:
            f.write(f'{layer_name}\n')
    f.close()
    
    print(f'[Info] selective model {filename} {num_of_bytes} bytes. {len(suspected_layers)} layers')
    demo(filename)
    
def demo(model_path):
    # Show Before Quantization
    infer = tf.keras.models.load_model(FLAGS.weights)
    res = infer(np.random.uniform(0, 1, size=(1,608,608,3)))
    print(f'res[0] {res[0].shape}')
    print(f'res[1] {res[1].shape}')
    
    logging.info(f'demo function. load model from {model_path}')
    interpreter = tf.lite.Interpreter(model_path=model_path)  # , experimental_preserve_all_tensors=True
    op_list = [op['op_name'] for op in interpreter._get_ops_details()]
    op_list = set(op_list)
    print(op_list)
    interpreter.allocate_tensors()
    ##############################
    # input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    # output = interpreter.tensor(299)
    # input().fill(3.)
    # interpreter.invoke()
    # print(f"inference tensor {output().shape} {output().dtype}")
    ##############################

    input_details = interpreter.get_input_details()
    print('input_details')
    for i in range(len(input_details)): print('\t',input_details[i])

    output_details = interpreter.get_output_details()
    print('output_details')
    for i in range(len(output_details)): print('\t',output_details[i])

    print('_get_full_signature_list', interpreter._get_full_signature_list())

    # Get tensor signature for certain tensor_idx  
    # interpreter._get_tensor_details(tensor_idx)


    input_shape = input_details[0]['shape']

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    print('input_data', input_data.shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    print("output_details[0]['index']", output_details[0]['index'])
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    print(output_data)
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass