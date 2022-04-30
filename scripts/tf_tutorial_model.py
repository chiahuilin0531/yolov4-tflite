import enum
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=3)(inputs)
    print(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(conv1)
    print(conv2)
    pooling = tf.keras.layers.GlobalAveragePooling2D()(conv2)
    print(pooling)
    feature = tf.keras.layers.Dense(10)(pooling)

    full_model = tf.keras.Model(inputs, feature)
    backbone = tf.keras.Model(inputs, conv2)
    activations = tf.keras.Model(conv2, feature)
    multi_output_model = tf.keras.Model(inputs, [feature, conv1, conv2])
    return full_model, backbone, activations, multi_output_model

if __name__=='__main__':
    batch = 4
    input_image = tf.ones((batch, 32, 32, 3))
    full_model, backbone, activations, multi_output_model = build_model()

    full_model_output=full_model(input_image)
    print(f'full_model_output: {full_model_output.shape}')

    backbone_output=backbone(input_image)
    print(f'backbone_output: {backbone_output.shape}')

    activations_output=activations(backbone_output)
    print(f'activations_output: {activations_output.shape}')

    print(tf.reduce_all(full_model_output==activations_output))

    multi_output = multi_output_model(input_image)
    for i, data in enumerate(multi_output):
        print(f'{i}th output: {data.shape} ')

