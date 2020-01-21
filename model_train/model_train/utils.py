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

# -------------------------------- activation utils --------------------------------

def chip_acti_fun(x):
    x = tf.nn.relu(x)
    x = 1-tf.exp(-2*x)
    return x

def activation(x, mode):
    if mode =='relu':
        out = tf.nn.relu(x)
    elif mode =='selu':
        out = tf.nn.selu(x)
    elif mode =='chip_relu':
        out = chip_acti_fun(x)
    elif mode == 'leaky_relu':
        out = tf.nn.leaky_relu(x, 0.1)
    elif mode == 'relu6':
        out = tf.nn.relu6(x)
    return out

# -------------------------------- dropout utils --------------------------------

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

    if mode=='relu' or 'leaky_relu':
        y=tf.nn.dropout(x, rate=dropout_prob)
    elif mode=='selu':
        y=dropout_selu(x, dropout_prob, training=True)
    elif mode=='chip_relu':
        # to be modified
        y=tf.nn.dropout(x, rate=dropout_prob)

    return y

# -------------------------------- primary conv. and fc layers --------------------------------

def conv_layer(x, kernel_size, stride, padding='SAME', mode='keras', name='c'):
    print("conv_layer({0} mode): kernel--{1}, stride--{2}, name--{3}, inp_shape--{4}".format(mode, str(kernel_size), str(stride), name, str(x.shape)))
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

def pointwise_conv_layer(x, input_channel, output_channel, name = 'pwise_c'):
    return conv_layer(x, [1,1,input_channel, output_channel], [1,1,1,1], 
                      padding='SAME', mode='tensorflow', name=name)

def depthwise_conv_layer(x, kernel_size, stride, padding='SAME', name = 'dwise_c'):
    '''
    kernel_size = [x, y, input_channel, 1]
    the 4th dim. must be 1
    '''
    print("ds_conv_layer:"+" "*20+" kernel--{0}, stride--{1}, name--{2}, inp_shape--{3}".format(str(kernel_size), str(stride), name, str(x.shape)))
    w_depthwise = tf.Variable(
             tf.random.truncated_normal(
              [kernel_size[0], kernel_size[1], kernel_size[2], 1],
              stddev = np.sqrt(2/(kernel_size[0]*kernel_size[1]))),name=name+'_w')
    b_depthwise = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[2]], 
                stddev=0.001),name=name+'_b')
    out_depthwise = tf.nn.depthwise_conv2d(x, w_depthwise, stride, padding='SAME') + b_depthwise
    return out_depthwise

def fc_layer(x, last_layer_element_count, unit_num, mode = 'keras', name='fc'):
    print("fc_layer({0} mode): input_unit--{1}, out_unit--{2}, name--{3}".format(mode, last_layer_element_count, unit_num, name))
    if mode =='tensorflow':
        w = tf.Variable(
                tf.random.truncated_normal(
                [last_layer_element_count, unit_num], 
                stddev=np.sqrt(2/last_layer_element_count)),
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

def batch_norm(x,momentum=0.99,scale=True,trainable=True, name='_bn'):
    print("batch_norm: momentum--{0}, trainable--{1}, name--{2}".format(momentum, trainable, name))
    return tf.layers.BatchNormalization(momentum=momentum,scale=scale,trainable=trainable, name=name)(x)



# -------------------------------- ResNet blocks --------------------------------

def identity_block_v1(x, kernel_size, stride, is_training,
                   padding='SAME', init_mode='selu', activation_mode='selu', name='b'):
    '''
    refer to ResNet yet BN is not adopted
    （Deep Residual Learning for Image Recognition.2015）
    https://arxiv.org/pdf/1512.03385.pdf
    '''
    print("identity_block_v1: name--{0}".format(name))
    inp = x 
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[2]], 
                   stride, padding='SAME', mode=init_mode, name=name+'_c1')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn1')
    x = activation(x, activation_mode)


    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=init_mode, name = name+'_c2')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn2')

    x = tf.add(inp, x)
    out = activation(x, activation_mode)
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
    #x = tf.layers.BatchNormalization(momentum=0.9,scale=True,trainable=is_training)(x)

    x = activation(x, mode)



    x = conv_layer(x, [kernel_size[0],kernel_size[1],expend_channel,expend_channel], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)

    x = activation(x, mode)


    x = conv_layer(x, [1,1,expend_channel,kernel_size[3]], 
                   stride, padding='SAME', mode=mode, name = name+'_c2')
    #x = tf.keras.layers.BatchNormalization()(x)

    x = activation(x, mode)

    
    out = tf.add(inp, x)

    return out

def conv_block_v1(x, kernel_size, stride, is_training, init_mode='selu', activation_mode='selu', name='conv_block'):
    print("conv_block: name--{0}".format(name))
    inp = x
    # main path
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=init_mode, name=name+'_c1')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn1')
    x = activation(x, activation_mode)

    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[3],kernel_size[3]], 
                   [1,1,1,1], padding='SAME', mode=init_mode, name=name+'_c2')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn2')
    

    # direct shortcut
    d = conv_layer(inp, kernel_size, 
                   [1,2,2,1], padding='SAME', mode=init_mode, name=name+'_c3')
    

    out = tf.add(d, x)
    out = activation(out, activation_mode)
    return out

def conv_block_v2(x, kernel_size, stride, is_training, init_mode='selu', activation_mode='selu', name='conv_block'):
    print("conv_block: name--{0}".format(name))

    # without shortcut
    x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[2],kernel_size[3]], 
                   stride, padding='SAME', mode=init_mode, name=name+'_c1')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn1')
    out = activation(x, activation_mode)

    #x = conv_layer(x, [kernel_size[0],kernel_size[1],kernel_size[3],kernel_size[3]], 
    #               [1,1,1,1], padding='SAME', mode=init_mode, name=name+'_c2')
    #x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn2')
    
    #out = activation(x, activation_mode)
    return out

# -------------------------------- Depthwise Separable conv. --------------------------------

def mobilenetv2_block_1(x, kernel_size, is_training, expension_factor=6, name='mobile_block1'):
    '''
    block_1: Stride=1 block
    '''
    print('-'*20 + "mobilenetv2_identity_block: name--{0}".format(name))
    inp = x
    x = pointwise_conv_layer(x, kernel_size[2], kernel_size[2]*expension_factor, name=name+'_pwise1')
    x = batch_norm(x, momentum=0.9,scale=True,trainable=is_training, name=name+'_bn1')
    x = activation(x, 'relu6')
    x = depthwise_conv_layer(x, [kernel_size[0], kernel_size[1], kernel_size[2]*expension_factor, 1],
                             [1,1,1,1], name=name + '_dwise')
    x = batch_norm(x, momentum=0.9,scale=True,trainable=is_training, name=name+'_bn2')
    x = activation(x, 'relu6')
    
    x = pointwise_conv_layer(x, kernel_size[2]*expension_factor, kernel_size[3], name=name+'_pwise2')
    out = tf.add(inp, x)
    return out

def mobilenetv2_block_2(x, kernel_size, is_training, expension_factor=2, name='mobile_block2'):
    '''
    block_2: Stride=2 block
    '''
    print('-'*20 + "mobilenetv2_direct_block: name--{0}".format(name))
    x = pointwise_conv_layer(x, kernel_size[2], kernel_size[2]*expension_factor, name=name+'_pwise1')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn1')
    x = activation(x, 'relu6')
    x = depthwise_conv_layer(x, [kernel_size[0], kernel_size[1], kernel_size[2]*expension_factor, 1],
                             [1,2,2,1], padding='SAME', name=name + '_dwise')
    x = batch_norm(x, momentum=0.99,scale=True,trainable=is_training, name=name+'_bn2')

    x = activation(x, 'relu6')
    x = pointwise_conv_layer(x, kernel_size[2]*expension_factor, kernel_size[3], name=name+'_pwise2')
    
    out = x
    return out


# -------------------------------- audio utils --------------------------------

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

# -------------------------------- image utils --------------------------------
# -------------------------------- other utils --------------------------------

def spase_to_onehot(label, class_num):
    # label.shape should be [batch, label]
    onehot_label = np.zeros([len(label), class_num])
    for i in range(len(label)):
        onehot_label[i][label[i]] = 1
    return onehot_label

class moving_average:
    def __init__(self, init_avg=0.5, beta=0.99):
        self.val = 0.
        self.beta = beta
        self.accumulate = init_avg
    def update(self, x):
        self.val = x
        self.accumulate = self.accumulate*self.beta + (1-self.beta)*self.val

import time
def get_current_time():
    t = time.localtime(time.time())
    local_time = str(t.tm_year)+'_'+str(t.tm_mon)+'_'+str(t.tm_mday)+'_'+str(t.tm_hour)
    return local_time

def count_conv_params_flops(conv_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = conv_layer.output.shape.as_list()
    n_cells_total = np.prod(out_shape[1:-1])

    n_conv_params_total = conv_layer.count_params()

    conv_flops = 2 * n_conv_params_total * n_cells_total

    if verbose:
        print("layer %s params: %s" % (conv_layer.name, "{:,}".format(n_conv_params_total)))
        print("layer %s flops: %s" % (conv_layer.name, "{:,}".format(conv_flops)))

    return n_conv_params_total, conv_flops

def count_dense_params_flops(dense_layer, verbose=1):
    # out shape is  n_cells_dim1 * (n_cells_dim2 * n_cells_dim3)
    out_shape = dense_layer.output.shape.as_list()
    n_cells_total = np.prod(out_shape[1:-1])

    n_dense_params_total = dense_layer.count_params()

    dense_flops = 2 * n_dense_params_total

    if verbose:
        print("layer %s params: %s" % (dense_layer.name, "{:,}".format(n_dense_params_total)))
        print("layer %s flops: %s" % (dense_layer.name, "{:,}".format(dense_flops)))

    return n_dense_params_total, dense_flops

def count_model_params_flops(model):
    total_params = 0
    total_flops = 0

    model_layers = model.layers

    for layer in model_layers:

        if any(conv_type in str(type(layer)) for conv_type in ['Conv1D', 'Conv2D', 'Conv3D']):
            params, flops = count_conv_params_flops(layer)
            total_params += params
            total_flops += flops
        elif 'Dense' in str(type(layer)):
            params, flops = count_dense_params_flops(layer)
            total_params += params
            total_flops += flops
        else:
            print("----------skippring layer: %s" % str(layer))
    
    total_mul = total_flops//2
    print('-------------------------------------------------------')
    print("total Param. (%s) : %s" % (model.name, "{:,}".format(total_params)))
    print("total Mul.  (%s) : %s" % (model.name, "{:,}".format(total_mul)))
    print("total FLOPs  (%s) : %s" % (model.name, "{:,}".format(total_flops)))

    return total_params, total_flops

def count_mobilenet_v2_param(stage = [3,3,3,3], input_channel = [16,32,64,128], expension = 3, kernel_size = [3, 3], is_rgb=True):
    if is_rgb:
        input_channel.insert(0, 3)
    else:
        input_channel.insert(0, 1)
    stage_len = len(stage)
    total_count = 0
    feature_mixed_channel = 256
    # input conv
    total_count = total_count + 3*3*input_channel[0]*input_channel[1]
    # block
    for s in range(1, len(stage)+1):
            # stride=2
            # point-wise
        total_count = total_count + 1*1*input_channel[s]*input_channel[s]*expension
            # dwise
        total_count = total_count + kernel_size[0]*kernel_size[1]*input_channel[s+1]*expension
            # point-wise
        total_count = total_count + 1*1*input_channel[s+1]*expension*input_channel[s+1]
        for i in range(stage[s-1]):
            # point-wise
            total_count = total_count + 1*1*input_channel[s+1]*input_channel[s+1]*expension
            # dwise
            total_count = total_count + kernel_size[0]*kernel_size[1]*input_channel[s+1]*expension
            # point-wise
            total_count = total_count + 1*1*input_channel[s+1]*expension*input_channel[s+1]
    
    # feature_mixed
    total_count = total_count + 1*1*input_channel[-1]*feature_mixed_channel
    # fc 
    total_count = total_count + feature_mixed_channel*2

    return total_count


