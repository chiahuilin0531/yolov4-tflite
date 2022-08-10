import tensorflow as tf
import cv2
import math
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
    net = common.convolutional(net, (3, 3,          3,   channels), downsample=True, activate=True, activate_type = 'leaky_relu', nl=None, prefix='ex_conv0')
    net = common.convolutional(net, (3, 3,   channels, 2*channels), downsample=True, activate=True, activate_type = 'leaky_relu', nl=None, prefix='ex_conv1')
    net = common.convolutional(net, (3, 3, 2*channels, 2*channels), downsample=True, activate=True, activate_type = 'leaky_relu', nl=None, prefix='ex_conv2')
    net = common.convolutional(net, (3, 3, 2*channels, 2*channels), downsample=True, activate=True, activate_type = 'leaky_relu', nl=None, prefix='ex_conv3')
    net = common.convolutional(net, (3, 3, 2*channels, 2*channels), downsample=True, activate=True, activate_type = 'leaky_relu', nl=None, prefix='ex_conv4')
    net = tf.reshape(net, [-1, 2048])
    features =        tf.keras.layers.Dense(64,         activation='leaky_relu', use_bias=True,  kernel_initializer='glorot_normal')(net)
    filter_features = tf.keras.layers.Dense(output_dim, activation=None,         use_bias=False, kernel_initializer='glorot_normal')(features)
    return filter_features

def DIP_FilterGraph(prossed_imgs, cnn_pp_params):
    prossed_imgs = ImprovedWhiteBalanceFilterGraph(prossed_imgs, cnn_pp_params)
    prossed_imgs = GammaFilterGraph(prossed_imgs, cnn_pp_params)
    prossed_imgs = ToneFilterGraph(prossed_imgs, cnn_pp_params)
    prossed_imgs = ContrastFilterGraph(prossed_imgs, cnn_pp_params)
    prossed_imgs = UsmFilterGraph(prossed_imgs, cnn_pp_params)
    return prossed_imgs

#####################################################################################

def ImprovedWhiteBalanceFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    # (batch, 3)
    used_params = cnn_pp_params[:, :3]
    
    '''
    # (batch, 1, 1, 3)
    used_params = tf.expand_dims(
        # (batch, 1, 3)
        tf.expand_dims(used_params, axis=1), 
    axis=1)
    '''

    #filter_parameters = filter_param_regressor(filter_features)
    log_wb_range = 0.5
    #mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    mask = tf.constant([[0,1,1]], dtype=tf.float32)
    # mask = np.array(((1, 0, 1)), dtype=np.float32).reshape(1, 3)

    # print(mask.shape)
    assert mask.shape == (1, 3)
    used_params = used_params * mask
    color_scaling = tf.exp(tanh_range(-log_wb_range, log_wb_range)(used_params))
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    color_scaling *= 1.0 / (
        1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2])[:, None]
    used_params = color_scaling
    

    #low_res_output = self.process(img, filter_parameters)
    prossed_imgs = prossed_imgs * used_params[:, None, None, :]
    
    return prossed_imgs

def GammaFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    #filter_features, mask_parameters = self.extract_parameters(img_features)
    # (batch, 3)
    used_params = cnn_pp_params[:, 3:4]
    
    '''
    # (batch, 1, 1, 3)
    used_params = tf.expand_dims(
        # (batch, 1, 3)
        tf.expand_dims(used_params, axis=1), 
    axis=1)
    '''

    #filter_parameters = filter_param_regressor(filter_features)
    #log_gamma_range = np.log(cfg.gamma_range)
    log_gamma_range = tf.math.log(cfg.gamma_range)
    used_params = tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(used_params))
    

    #low_res_output = self.process(img, filter_parameters)
    param_1 = tf.tile(used_params, [1, 3])
    prossed_imgs = tf.pow(tf.maximum(prossed_imgs, 0.001), param_1[:, None, None, :])
    
    return prossed_imgs

def ToneFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    #filter_features, mask_parameters = self.extract_parameters(img_features)
    # (batch, 3)
    used_params = cnn_pp_params[:, 4:8]
    '''
    # (batch, 1, 1, 4)
    used_params = tf.expand_dims(
        # (batch, 1, 4)
        tf.expand_dims(used_params, axis=1), 
    axis=1)
    '''

    #filter_parameters = filter_param_regressor(filter_features)
    #cfg.curve_steps = 4
    tone_curve = tf.reshape(used_params, shape=(-1, 1, cfg.curve_steps))[:, None, None, :] #(14,4) -> (14,1,4) -> (14,1,1,4)
    #cfg.tone_curve_range = (0.5, 2)
    used_params = tanh_range(*cfg.tone_curve_range)(tone_curve) #(14,1,1,4)

    #low_res_output = self.process(img, filter_parameters)
    # img = tf.minimum(img, 1.0)
    tone_curve = used_params
    tone_curve_sum = tf.reduce_sum(tone_curve, axis=4) + 1e-30 #(14,1)
    total_image = prossed_imgs * 0
    for i in range(cfg.curve_steps):
        total_image += tf.clip_by_value(prossed_imgs - 1.0 * i / cfg.curve_steps, 0, 1.0 / cfg.curve_steps) \
                        * used_params[:, :, :, :, i]
    total_image *= cfg.curve_steps / tone_curve_sum
    prossed_imgs = total_image
    
    return prossed_imgs

def ContrastFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    #filter_features, mask_parameters = self.extract_parameters(img_features)
    # (batch, 3)
    used_params = cnn_pp_params[:, 12:13]
    
    '''
    # (batch, 1, 1, 3)
    used_params = tf.expand_dims(
        # (batch, 1, 3)
        tf.expand_dims(used_params, axis=1), 
    axis=1)
    '''

    #filter_parameters = filter_param_regressor(filter_features)
    used_params = tf.tanh(used_params)

    #low_res_output = self.process(img, filter_parameters)
    # img = tf.minimum(img, 1.0)
    luminance = tf.minimum(tf.maximum(rgb2lum(prossed_imgs), 0.0), 1.0)
    contrast_lum = -tf.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = prossed_imgs / (luminance + 1e-6) * contrast_lum
    prossed_imgs = lerp(prossed_imgs, contrast_image, used_params[:, :, None, None])
    
    return prossed_imgs
 
def UsmFilterGraph(prossed_imgs, cnn_pp_params):
    """
    prossed_imgs: (batch, 608, 608, 3)
    cnn_pp_params: (batch, 14)
    """
    # (batch, 1)
    used_params = cnn_pp_params[:, 13:14]
    used_params = tanh_range(*cfg.usm_range)(used_params)
    
    #low_res_output = self.process(img, filter_parameters)
    def make_gaussian_2d_kernel(sigma, dtype=tf.float32):
        radius = 12
        x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
        k = tf.exp(-0.5 * tf.square(x / sigma))
        k = k / tf.reduce_sum(k)
        return tf.expand_dims(k, 1) * k

    # kernel_i = (25,25,1,1)
    kernel_i = make_gaussian_2d_kernel(5)
    kernel_i = tf.tile(kernel_i[..., tf.newaxis, tf.newaxis] / 3.0, [1,1,3,3])
    
    pad_w = (25 - 1) // 2
    padded = tf.pad(prossed_imgs, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    # outputs = []
    # for channel_idx in range(3):
    #     data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
    #     data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], 'VALID')
    #     outputs.append(data_c)

    # output = tf.concat(outputs, axis=3)
    output = tf.nn.conv2d(padded, kernel_i, 1, 'VALID')
    prossed_imgs = (prossed_imgs - output) * used_params[:, tf.newaxis, tf.newaxis, :] + prossed_imgs
    
    return prossed_imgs

def tanh01(x):
  return tf.tanh(x) * 0.5 + 0.5

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







