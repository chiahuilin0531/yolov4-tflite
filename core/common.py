#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow_addons as tfa


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class InstanceNormalization(tfa.layers.InstanceNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x)


NORMALIZATION = {
    'BatchNorm': BatchNormalization,
    'InstanceNorm': tfa.layers.InstanceNormalization,
    # 'LayerNorm': tfa.layers.LayerNormalization,
}

def convolutional(input_layer, filters_shape, downsample=False, activate=True, nl='BatchNorm', activate_type='relu', prefix=None):
    any_nl = nl is not None
    if prefix==None: 
        conv_name = None
        bn_name = None
        pad_name = None
    else:
        conv_name = prefix + '_conv2d'
        bn_name = (prefix + '_batch_normalization')
        pad_name = (prefix + '_zero_padding2d')
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name=pad_name)(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not any_nl, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.random_normal_initializer(stddev=0.0001), name=conv_name)(input_layer)

    if any_nl: conv = NORMALIZATION[nl](name=bn_name)(conv)
    if activate == True:
        # support activation: relu, relu6, swish
        conv = tf.keras.layers.Activation(activate_type)(conv)
        # if activate_type == "leaky":
        #     conv = tf.nn.leaky_relu(conv, alpha=0.1)
        # elif activate_type == 'relu':
        #     # activate_type = 'swish'# conv = tf.nn.relu6(conv)
        #     # conv = tf.nn.relu(conv)
        #     conv = tf.keras.layers.Activation(activate_type)(conv)
        # elif activate_type == "mish":
        #     conv = mish(conv)

    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class MishLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MishLayer, self).__init__()
    def call(self, x, training=False):
        return x * tf.math.tanh(tf.math.softplus(x))

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    # convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    # return convs[group_id]
    convs = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=groups, axis=-1))(input_layer)
    return convs[group_id]

def upsample(input_layer):
    # return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
    convs = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear'))(input_layer)
    return convs
@tf.custom_gradient
def grad_reverse(x):
    # write your gradient reversal operation
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradientReversal(tf.keras.layers.Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda=1.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def call(self, x, mask=None):
        return grad_reverse(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))