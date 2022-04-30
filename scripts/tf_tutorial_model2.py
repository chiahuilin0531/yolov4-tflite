import enum
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_tuple_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=3)(inputs)
    conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(conv1)
    pooling = tf.keras.layers.GlobalAveragePooling2D()(conv2)
    feature = tf.keras.layers.Dense(10)(pooling)

    multi_output_model = tf.keras.Model(inputs, [feature, conv1, conv2])
    return multi_output_model

def build_dict_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    conv1 = tf.keras.layers.Conv2D(filters=4, kernel_size=3)(inputs)
    conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(conv1)
    pooling = tf.keras.layers.GlobalAveragePooling2D()(conv2)
    feature = tf.keras.layers.Dense(10)(pooling)

    multi_output_model = tf.keras.Model(inputs, {
        'my_feature': feature,
        'my_conv1': conv1,
        'my_conv2': conv2
    })
    return multi_output_model

if __name__=='__main__':
    batch = 4
    input_image = tf.ones((batch, 32, 32, 3))

    multi_tuple_output_model = build_tuple_model()
    multi_dict_output_model = build_dict_model()

    multi_tuple_output = multi_tuple_output_model(input_image)
    for i, data in enumerate(multi_tuple_output):
        print(f'{i}th output: {data.shape} ')
    
    multi_dict_output = multi_dict_output_model(input_image)
    for i, (key,data) in enumerate(multi_dict_output.items()):
        print(f'{key} output: {data.shape} ')

