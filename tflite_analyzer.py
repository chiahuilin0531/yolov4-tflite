import tensorflow as tf


MODEL_PATH = './checkpoints/0923_F3/tflite_0060/int8_model_33_layer_st82_end115.tflite'


tf.lite.experimental.Analyzer.analyze(
    model_path=MODEL_PATH, gpu_compatibility=True
)