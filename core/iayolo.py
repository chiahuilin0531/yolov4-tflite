import tensorflow as tf
import cv2
import math
import tensorflow_addons as tfa
from core.config import cfg
import core.common as common
cfg.num_filter_parameters = 14
cfg.wb_begin_param = 0#3
cfg.gamma_begin_param = 3#1
cfg.tone_begin_param = 4#4
cfg.contrast_begin_param = 12#1
cfg.usm_begin_param = 13#1

cfg.curve_steps = 4
cfg.gamma_range = 2.5
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.5, 1.0)
cfg.usm_range = (0.0, 2.5)
cfg.contrast_range = (0.0, 1.0)


def CNNPP(net):
    output_dim = 14
    channels = 16
    act='swish'  # 'leaky_relu'
    nl = 'BatchNorm' # 'BatchNorm'   # None  
    l2=5e-3 # 1e-3 # 5e-4
    net = tfa.layers.InstanceNormalization()(net)
    net = common.convolutional(net, (3, 3,          3,   channels), downsample=True, activate=True, activate_type = act, nl=nl, prefix='ex_conv0', l2_reg=l2)
    net = common.convolutional(net, (3, 3,   channels, 2*channels), downsample=True, activate=True, activate_type = act, nl=nl, prefix='ex_conv1', l2_reg=l2)
    net = common.convolutional(net, (3, 3, 2*channels, 2*channels), downsample=True, activate=True, activate_type = act, nl=nl, prefix='ex_conv2', l2_reg=l2)
    net = common.convolutional(net, (3, 3, 2*channels, 2*channels), downsample=True, activate=True, activate_type = act, nl=nl, prefix='ex_conv3', l2_reg=l2)
    net = common.convolutional(net, (3, 3, 2*channels, 2*channels), downsample=True, activate=True, activate_type = act, nl=nl, prefix='ex_conv4', l2_reg=l2)

    
    # net =        tfa.layers.InstanceNormalization()(net)
    implementation=2
    
    # Old Implementation Use Dense Layer
    if implementation==0:
        net = tf.reshape(net, [-1, 2048])
        features =        tf.keras.layers.Dense(64,         activation=act,     use_bias=True, kernel_initializer='glorot_normal', \
            name='ex_conv5', kernel_regularizer=tf.keras.regularizers.l2(5e-3))(net) # 5e-4
        features = tf.keras.layers.BatchNormalization(name='ex_batch')(features)
        filter_features = tf.keras.layers.Dense(output_dim, activation=None,    use_bias=True, kernel_initializer='glorot_normal', \
            name='ex_conv6')(features)
    
    # New Implementation
    elif implementation==1:
        net = tf.reshape(net, [-1, 1, 1, 2048])
        features =          common.convolutional(net,      (1, 1, 2048, 64),       activate=True, activate_type = act,  nl=nl, prefix='ex_conv5', l2_reg=l2)
        filter_features =   common.convolutional(features, (1, 1, 64, output_dim), activate=False,                      nl=None, prefix='ex_conv6', l2_reg=l2)
        filter_features = tf.reshape(filter_features, [-1, 14])
    
    # 2nd Implementation Use Conv2D Layer
    elif implementation==2:
        net = tf.reshape(net, [-1, 1, 1, 2048])
        features = tf.keras.layers.Conv2D(64, (1,1), kernel_initializer='glorot_normal', \
            kernel_regularizer=tf.keras.regularizers.l2(l2), use_bias=False, name='ex_conv5')(net)
        features = tf.keras.layers.BatchNormalization(name='ext_batch5')(features)
        features = tf.keras.layers.Activation('swish', name='ext_act5')(features)
        
        filter_features = tf.keras.layers.Conv2D(output_dim, (1,1), kernel_initializer='glorot_normal', \
            use_bias=True, name='ex_conv6')(features)    
        filter_features = tf.reshape(filter_features, [-1, 14])


    return filter_features

def DIP_FilterGraph(prossed_imgs, cnn_pp_params):
    graph_list = [ImprovedWhiteBalanceFilterGraph, GammaFilterGraph, ToneFilterGraph, ContrastFilterGraph, UsmFilterGraph]
    processed_list = []
    for graph in graph_list:
        prossed_imgs = graph(prossed_imgs, cnn_pp_params)
        processed_list.append(prossed_imgs)

    return prossed_imgs, processed_list

#####################################################################################

# VV
def ImprovedWhiteBalanceFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
        use 3 parameter(0,1,2)
    """
    ###########################################
    # Parameter Regression Part
    ###########################################
    # (batch, 3)
    used_params = cnn_pp_params[:, :3]
    log_wb_range = 0.5
    mask = tf.constant([[0,1,1]], dtype=tf.float32)

    assert mask.shape == (1, 3)
    # Use only green and blue color to do White Balance
    used_params = used_params * mask
    # color_scaling=(batch, 3), value_range=(-0.5, 0.5)
    color_scaling = tf.exp(tanh_range(-log_wb_range, log_wb_range)(used_params))
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    color_scaling *= 1.0 / (1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2])[:, None]
    used_params = color_scaling
    

    ###########################################
    # Image Processing Part
    ###########################################    
    prossed_imgs = prossed_imgs * used_params[:, None, None, :]
    
    return prossed_imgs

# VV 
# Extensive Operator Exp
def GammaFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
        use 1 parameter(3)
    """
    ###########################################
    # Parameter Regression Part
    ###########################################
    # (batch, 1)
    used_params = cnn_pp_params[:, 3:4] 
    log_gamma_range = tf.math.log(cfg.gamma_range) # gamma_range=2.5
    # value_range=(-0.6, 1.5)
    used_params = tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(used_params))
    

    ###########################################
    # Image Processing Part
    ########################################### 
    # shape=(batch, 3)
    # param_1 = tf.tile(used_params, [1, 3])
    # Prevent Using TILE operation !!
    param_1 = tf.concat([used_params, used_params, used_params], axis=-1)
    prossed_imgs = tf.pow(tf.maximum(prossed_imgs, 0.001), param_1[:, None, None, :])
    
    return prossed_imgs

# VV
# Extensive Operator SUB (prossed_imgs - 1.0 * i / cfg.curve_steps), 5 Quantization
def ToneFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
        use 4 parameter(4,5,6,7)
    """
    ###########################################
    # Parameter Regression Part
    ###########################################
    # (batch, 4)
    used_params = cnn_pp_params[:, 4:8]
    # tone_curve = tf.reshape(used_params, shape=(-1, 1, cfg.curve_steps))[:, None, None, :] #(batch, 4) -> (batch, 1, 4) -> (batch, 1, 1, 1, 4)
    # tone_curve = tf.reshape(used_params, shape=(-1, 1, 1, 1, cfg.curve_steps))
    tone_curve = used_params
    # used_params=(batch, 1, 1, 1, 4), value_range=(0.5,2)
    used_params = tanh_range(*cfg.tone_curve_range)(tone_curve) #(batch, 1, 1, 1, 4)

    ###########################################
    # Image Processing Part
    ###########################################
    tone_curve = used_params
    tone_curve_sum = tf.reduce_sum(tone_curve, axis=-1) + 1e-30 #(batch, 1, 1, 1)
    tone_curve_sum = tf.reshape(tone_curve_sum, (-1, 1, 1, 1))
    total_image = prossed_imgs * 0
    for i in range(cfg.curve_steps):
        tone_step = tf.reshape(used_params[:, i], (-1, 1, 1, 1))
        total_image += tf.clip_by_value(prossed_imgs - 1.0 * i / cfg.curve_steps, 0, 1.0 / cfg.curve_steps) * tone_step
    total_image *= cfg.curve_steps / tone_curve_sum
    prossed_imgs = total_image
    
    return prossed_imgs

# VV
# Extensive Operator Cos
# Extensive Operator Div (contrast_image = prossed_imgs / (luminance + 1e-6) * contrast_lum) 
# both nominator and denominator require to be dequantize 
def ContrastFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    ###########################################
    # Parameter Regression Part
    ###########################################
    # (batch, 1)
    used_params = cnn_pp_params[:, 12:13]
    used_params = tf.tanh(used_params)

    ###########################################
    # Image Processing Part
    ###########################################
    luminance = tf.minimum(tf.maximum(rgb2lum(prossed_imgs), 0.0), 1.0)
    contrast_lum = -tf.cos(math.pi * luminance) * 0.5 + 0.5
    # contrast_lum = -tf.sin(math.pi * luminance + 1.570796) * 0.5 + 0.5
    
    contrast_image = prossed_imgs / (luminance + 1e-6) * contrast_lum
    prossed_imgs = lerp(prossed_imgs, contrast_image, used_params[:, :, None, None])
    
    return prossed_imgs

# VV
def UsmFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    ###########################################
    # Parameter Regression Part
    ###########################################
    # (batch, 1)
    used_params = cnn_pp_params[:, 13:14]
    used_params = tanh_range(*cfg.usm_range)(used_params)
    
    ###########################################
    # Image Processing Part
    ###########################################
    radius = 6
    def make_gaussian_2d_kernel(sigma, dtype=tf.float32):
        # x=(radius, )
        x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
        # k=(radius, )
        k = tf.exp(-0.5 * tf.square(x / sigma))
        # normalize k
        k = k / tf.reduce_sum(k)
        return tf.expand_dims(k, 1) * k

    # kernel_i = (25,25,1,1)
    kernel_i = make_gaussian_2d_kernel(5)
    # padded = (b, 632, 632, 3)
    pad_w = (radius*2+1 - 1) // 2
    # padded = tf.pad(prossed_imgs, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='CONSTANT', value=0)
    padded = tf.keras.layers.ZeroPadding2D(padding=(pad_w, pad_w))(prossed_imgs)
    
    implementation=3
    if implementation == 1:
        kernel_i = tf.tile(kernel_i[..., tf.newaxis, tf.newaxis], [1,1,3,1])

        output = tf.nn.depthwise_conv2d(padded, kernel_i, [1,1,1,1], 'VALID')
    elif implementation == 2:
        kernel_i = tf.tile(kernel_i[..., tf.newaxis, tf.newaxis], [1,1,3,3]).numpy()
        kernel_i[:, :, [1,2], 0] = 0
        kernel_i[:, :, [0,2], 1] = 0
        kernel_i[:, :, [0,1], 2] = 0
        kernel_i = tf.constant(kernel_i)
        output = tf.nn.conv2d(padded, kernel_i, [1,1,1,1], 'VALID')
    elif implementation == 3:
        # this implementation require radius=6 gaussian filter, or the memory would out of range
        kernel_i = kernel_i[..., tf.newaxis, tf.newaxis]
        data_c1 = tf.nn.conv2d(padded[:, :, :, 0:1], kernel_i, [1, 1, 1, 1], 'VALID')
        data_c2 = tf.nn.conv2d(padded[:, :, :, 1:2], kernel_i, [1, 1, 1, 1], 'VALID')
        data_c3 = tf.nn.conv2d(padded[:, :, :, 2:3], kernel_i, [1, 1, 1, 1], 'VALID')
        output = tf.concat([data_c1, data_c2, data_c3], axis=3)
    
    extend_params = used_params[:, tf.newaxis, tf.newaxis, :]
    prossed_imgs = (prossed_imgs - output) * extend_params + prossed_imgs
    # # Implementation
    USM_Implementation=2
    if USM_Implementation==1:
        pass
    elif USM_Implementation==2:
        prossed_imgs = tf.clip_by_value(prossed_imgs, 0.0, 1.1)
    return prossed_imgs

def tanh01(x):
    # const_val = tf.ones(x.shape) * 0.5 # ValueError: Cannot convert a partially known TensorShape (None, 3) to a Tensor.
    # const_val = tf.ones((1,1,1,1,4)) * 0.5
    const_val = tf.ones((1, *x.shape[1:])) * 0.5
    
    return tf.tanh(x) * const_val + const_val

def tanh_range(l, r, initial=None):

    def get_activation(left, right, initial):

        def activation(x):
            if initial is not None:
                bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)

def rgb2lum(image):
    image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
    return image[:, :, :, None]

def lerp(a, b, l):
    return (1 - l) * a + l * b

def lrelu(x, leak=0.2, name="lrelu"):
    f1 = 0.5 * (1 + leak) # 0.6
    f2 = 0.5 * (1 - leak) # 0.4
    return f1 * x + f2 * abs(x)





