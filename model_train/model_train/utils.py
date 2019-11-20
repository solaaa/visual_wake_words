from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import random
#testing possible chip activation function
def chip_acti_fun(x):
    x = tf.nn.relu(x)
    x = 1-tf.exp(-2*x)
    return x


def conv_layer(x, kernel_size, stride, padding='SAME', mode='keras', name='c'):
    if mode == 'tensorflow':
        #he_normal
        w = tf.Variable(
             tf.random.truncated_normal(
              kernel_size,
              stddev=np.sqrt( 2/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
             name=name+'_w')
        b = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=0.001),
             name=name+'_b')
        c_out = tf.nn.conv2d(x, w, stride, padding) + b
    elif mode == 'keras':
        c_out = tf.keras.layers.Conv2D(kernel_size[3],
                                       [kernel_size[0], kernel_size[1]],
                                       [stride[1], stride[2]],
                                       padding=padding)(x)
    elif mode == 'selu':
        w = tf.Variable(
             tf.random.truncated_normal(
              kernel_size,
              stddev = np.sqrt( 1/(kernel_size[0]*kernel_size[1]*kernel_size[2]) )),
             name=name+'_w')
        b = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=0.001),
                 name=name+'_b')
        c_out = tf.nn.conv2d(x, w, stride, padding) + b
    return c_out

def fc_layer(x, last_layer_element_count, unit_num, mode = 'keras', name='fc'):
    if mode =='tensorflow':
        w = tf.Variable(
                tf.random.truncated_normal(
                [last_layer_element_count, unit_num], stddev=0.01),
                name = name+'_w')
        b = tf.Variable(
                tf.random.truncated_normal([unit_num], stddev=0.01),
                name = name+'_b')

        fc_out = tf.matmul(x, w) + b
    elif mode == 'keras':
        fc_out = tf.keras.layers.Dense(unit_num)(x)
    elif mode == 'selu' :
        w = tf.Variable(
             tf.random.truncated_normal(
                [last_layer_element_count, unit_num], 
                stddev=np.sqrt(1/last_layer_element_count)),
                name=name+'_w')
        b = tf.Variable(
             tf.random.truncated_normal(
                [unit_num], 
                stddev=0),
                name=name+'_b')

        fc_out = tf.matmul(x, w) + b
    return fc_out

def identity_block_v1(x, kernel_size, stride, 
                   padding='SAME', init_mode='selu', activation_mode='selu', name='b'):
    '''
    refer to ResNet yet BN is not adopted
    （Deep Residual Learning for Image Recognition.2015）
    https://arxiv.org/pdf/1512.03385.pdf
    '''
    inp = x # channel=32
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[2]], 
                   stride, padding='SAME', mode=init_mode, name=name+'_c1')
    x = tf.keras.layers.BatchNormalization()(x)
    #x = x * 0.00390625
    x = activation(x, activation_mode)
    #x = x * 256


    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=init_mode, name = name+'_c2')
    x = tf.keras.layers.BatchNormalization()(x)
    #x = x * 0.00390625
    x = activation(x, activation_mode)
    #x = x * 256
    out = tf.add(inp, x) * 1
    return out


def group_conv_block(x, group, kernel_size, stride, dropout_prob, is_training, 
                   padding='SAME', mode='selu', name='b'):
    '''
    ' ResNeXt: https://arxiv.org/pdf/1611.05431.pdf
    '''
    inp = x
    channel_per_group = int(kernel_size[2]/group)
    x_group = tf.split(x, group, -1)
    for i in range(group):
        x_group[i]=conv_layer(x_group[i], 
                                    [kernel_size[0],kernel_size[1],channel_per_group,channel_per_group], 
                                    stride, padding='SAME', mode=mode, name=name+'_g%d_c1'%(i))
        x_group[i] = activation(x_group[i]* 0.00390625, mode) * 256

        #x_group[i]=conv_layer(x_group[i], 
        #                            [kernel_size[0],kernel_size[1],channel_per_group,channel_per_group], 
        #                            stride, padding='SAME', mode=mode, name=name+'g%d_c2'%(i))
        #x_group[i] = activation(x_group[i]* 0.00390625, mode) * 256
    
    x = tf.concat([i for i in x_group], axis=-1)
    x=conv_layer(x, 
                    [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[2]], 
                    stride, padding='SAME', mode=mode, name=name+'_c2')
    x = activation(x* 0.00390625, mode) * 256
    out = tf.add(inp, x) * 0.5

    return out


def identity_block_v2(x, kernel_size, stride, dropout_prob, is_training, 
                   padding='SAME', mode='selu', name='b'):
    '''
    refer to ResNet yet BN is not adopted
    （Deep Residual Learning for Image Recognition.2015）
    https://arxiv.org/pdf/1512.03385.pdf
    '''
    expend_channel = 128
    inp = x # channel=32
    x = conv_layer(x, [1,1,kernel_size[2],expend_channel], 
                   stride, padding='SAME', mode=mode, name=name+'_c1')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = x * 0.00390625
    x = activation(x, mode)
    x = x * 256


    x = conv_layer(x, [kernel_size[0],kernel_size[1],expend_channel,expend_channel], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = x * 0.00390625
    x = activation(x, mode)
    x = x * 256

    x = conv_layer(x, [1,1,expend_channel,kernel_size[3]], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)
    x = x * 0.00390625
    x = activation(x, mode)
    x = x * 256
    
    out = tf.add(inp, x) * 0.5

    return out

def conv_block(x, kernel_size, stride, dropout_prob, is_training, 
                   padding='SAME', mode='selu'):
    inp = x 
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=mode)
    x = activation(x, mode)

    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[3],kernel_size[3]], 
                   stride, padding='SAME', mode=mode)
    x = activation(x, mode)


    # direct shortcut
    d = conv_layer(inp, [1,1,kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=mode)
    d = activation(d, mode)

    out = tf.add(d, x)
    #out = tf.nn.selu(out)
    return out


def depthwise_seperable_conv(x, kernel_size, stride):
    
    bn = 0

    w_depthwise = tf.Variable(
             tf.random.truncated_normal(
              [kernel_size[0], kernel_size[1], kernel_size[2], 1],
              stddev = np.sqrt( 2/(kernel_size[0]*kernel_size[1]*kernel_size[2]) )))
    b_depthwise = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[2]], 
                stddev=0.001))
    w_pointwise = tf.Variable(
             tf.random.truncated_normal(
              [1,1,kernel_size[2],kernel_size[3]],
              stddev = np.sqrt( 2/(kernel_size[0]*kernel_size[1]*kernel_size[2]) )))
    b_pointwise = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=0.001))

    out_depthwise = tf.nn.depthwise_conv2d(x, w_depthwise, stride, padding='SAME') + b_depthwise
    if bn:
        out_depthwise = tf.keras.layers.BatchNormalization()(out_depthwise)
    out_depthwise = tf.nn.relu(out_depthwise)

    out_pointwise = tf.nn.conv2d(out_depthwise, w_pointwise, [1,1,1,1], padding='SAME') + b_pointwise
    if bn:
        out_pointwise = tf.keras.layers.BatchNormalization()(out_pointwise)
    #out_pointwise = tf.nn.relu(out_pointwise)
    return out_pointwise

def ds_identity_block(x, kernel_size, stride, padding='SAME', mode='selu'):
    inp = x
    x = depthwise_seperable_conv(x, kernel_size, stride)
    x = depthwise_seperable_conv(x, kernel_size, stride)
    out = tf.add(inp, x)
    return out

def activation(x, mode):
    if mode=='relu':
        out = tf.nn.relu(x)
    elif mode=='selu':
        out = tf.nn.selu(x)
    elif mode=='chip_relu':
        out = chip_acti_fun(x)

    return out


def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling.
        1. Self-Normalizing Neural Networks
        https://arxiv.org/pdf/1706.02515.pdf
        https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    """

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        #keep_prob = rate # original code
        keep_prob = 1.0 - rate # rate means dropout_prob, keep_prob should be 1-rate.
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))




def dropout_with_mode(x, dropout_prob, mode):

    if mode=='relu':
        y=tf.nn.dropout(x, rate=dropout_prob)
    elif mode=='selu':
        y=dropout_selu(x, dropout_prob, training=True)
    elif mode=='chip_relu':
        #to be modified
        y=tf.nn.dropout(x, rate=dropout_prob)

    return y

def get_mask_par(x_size, rate):
    # build a masking 0-1 matrix
    mask_type = '1'
    ones = np.ones(x_size)
    # 1
    MAX_FREQ_MASK = 5
    MAX_TIME_MASK = 10

    # 2
    #MAX_TIME_MASK = int(x_size[1]//4)
    #MIN_TIME_MASK = int(x_size[1]//6)

    aug_num = int(x_size[0]*rate)
    random_index_mask = random.sample(range(0, x_size[0]), aug_num)
    #print(random_index_mask) # wrong: only run 1 time
    for i in random_index_mask:
        if mask_type == '1': # google method
            start_t = np.random.randint(0,x_size[1]-MAX_TIME_MASK)
            strat_f = np.random.randint(0,x_size[2]-MAX_FREQ_MASK)
            len_t = np.random.randint(2, MAX_TIME_MASK)
            len_f = np.random.randint(2, MAX_FREQ_MASK)
            ones[i][start_t:start_t+len_t, :] = 0.
            ones[i][:, strat_f:strat_f+len_f] = 0.
        else:
            # mask a whole freq. part from left or right
            rand_direct = np.random.randint(0, 2)
            len_t = np.random.randint(MIN_TIME_MASK, MAX_TIME_MASK)
            if rand_direct == 0: # left
                ones[i][0:len_t, :] = 0.
            else: # right
                ones[i][x_size[1]-len_t:len_t, :] = 0.
    return ones

def get_warp_par(x_size, rate):
    time_middle = x_size[1]//2
    MAX_W = 5 
    aug_num = int(x_size[0]*rate)
    w = np.random.randint(-MAX_W,MAX_W+1, [x_size[0],]) # left or right
    random_index_not_warp = random.sample(range(0, x_size[0]), x_size[0] - aug_num)
    w[random_index_not_warp] = 0
    boundary = np.array([[[0,0], [0, x_size[2]-1], [x_size[1]-1,0], [x_size[1]-1, x_size[2]-1]]])
    
    # to build the source and dest array for sparse_image_warp()
    s = np.array([[[time_middle-w[0], 0],[time_middle-w[0], x_size[2]-1]]])
    s = np.concatenate([s, boundary], axis=1)

    d = np.array([[[time_middle+w[0], 0],[time_middle+w[0], x_size[2]-1]]])
    d = np.concatenate([d, boundary], axis=1)
    for i in range(1, x_size[0]):
        temp_s = np.array([[[time_middle-w[i], 0],[time_middle-w[i], x_size[2]-1]]])
        temp_s = np.concatenate([temp_s, boundary], axis=1)
        s = np.concatenate([s, temp_s], axis=0)
        
        temp_d = np.array([[[time_middle+w[i], 0],[time_middle+w[i], x_size[2]-1]]])
        temp_d = np.concatenate([temp_d, boundary], axis=1)
        d = np.concatenate([d, temp_d], axis=0)
    return s, d

def mask(x, mask_matrix):
    #ones_tf = tf.constant(ones, dtype=tf.float32)
    x_mask = x * mask_matrix
    return x_mask

def time_warp(x, x_size, s, d):
    #s_tf = tf.constant(s, dtype=tf.float32)
    #d_tf = tf.constant(d, dtype=tf.float32)
    x = tf.reshape(x, shape=[x_size[0], x_size[1], x_size[2], 1])
    x_warp = tf.contrib.image.sparse_image_warp(x, s, d)
    x_warp = tf.reshape(x_warp[0], shape = x_size)
    return x_warp

def data_augment(x, x_size, mask_matrix, warp_source, warp_dest):
    '''
    ' param.:
    '  rate: the percentage of x to be augmented
    '''
    # default x_size = [128, 65, 40] in current code
    x = tf.reshape(x, shape=x_size)
    x_warp = time_warp(x, x_size, warp_source, warp_dest)
    #x_warp = x
    x_mask = mask(x_warp, mask_matrix)
    
    return x_mask


