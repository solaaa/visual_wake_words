from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from utils import conv_block, mobilenetv2_block_1, mobilenetv2_block_2, data_augment, group_conv_block, conv_layer, fc_layer, identity_block_v2,identity_block_v1, dropout_selu, conv_block, activation, dropout_with_mode


def create_model(input_batch, model_setting, model_architecture):
    if model_architecture == 'resnet':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return resnet(input_batch, model_setting, dropout_prob_ph)
    if model_architecture == 'mobile_net_v2':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return mobile_net_v2(input_batch, model_setting, dropout_prob_ph)
    if model_architecture == 'simple_convnet':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return simple_convnet(input_batch, model_setting, dropout_prob_ph)
    else: 
        raise Exception('model_architecture is not existing')

def resnet(input_batch, model_setting, dropout_prob_ph):
    '''
    input_batch: shape = [-1, height(240), width(320), 1]
    model_setting:
        training_layer_init_mode
        activation_mode
    '''
    # 1. init
    init_mode = model_setting['training_layer_init_mode']
    activation_mode = model_setting['activation_mode']
    
    dropout_prob = model_setting['dropout_prob']

    image_resolution_str = model_setting['image_resolution'].split(' ')
    image_resolution = [int(i) for i in image_resolution_str]
    input_height = image_resolution[0]
    input_width = image_resolution[1]

    input_channel = [16,16,32,64]

    stage = [1,2,2,1] # each stage includes N block

    if model_setting['color_mode'] == 'rgb':
        input_channel.insert(0,3)
    elif model_setting['color_mode'] == 'gray':
        input_channel.insert(0,1)

        
    # 2.0
    c0 = conv_layer(input_batch, 
                    [3, 3, input_channel[0], input_channel[1]],
                    [1,1,1,1],
                    mode=init_mode,
                    name='input_layer_c0')
    #c0 = tf.layers.BatchNormalization(momentum=0.9,scale=False)(c0)
    c0 = activation(c0, activation_mode)
    c0=dropout_with_mode(c0, dropout_prob_ph, activation_mode)
    di = c0

    # 2.1 residual
    for s in range(1, len(stage)+1):
        print(di.shape, s, input_channel[s])
        print('-----------------------------')
        for i in range(1, stage[s-1]+1):
            bi = identity_block_v1(di, [3,3,input_channel[s],input_channel[s]],[1,1,1,1],
                                   init_mode=init_mode, 
                                   activation_mode=activation_mode, 
                                   name='stage%d_block1%d'%(s,i))
            di=dropout_with_mode(bi, dropout_prob_ph, activation_mode)
            #di = bi
        
        if s < len(stage):
            print(di.shape, s, input_channel[s], input_channel[s+1])
            di = conv_block(di, [3,3,input_channel[s],input_channel[s+1]], [1,2,2,1], 
                            init_mode=init_mode, activation_mode=activation_mode)
            di=dropout_with_mode(di, dropout_prob_ph, activation_mode)

    # 3. global avg_pool
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(di)
    logits = fc_layer(avg_pool, input_channel[-1], 2, mode=init_mode, name = 'output_fc')
    softmax_prob = tf.keras.activations.softmax(logits)
    
    return logits, softmax_prob, dropout_prob_ph

def mobile_net_v2(input_batch, model_setting, dropout_prob_ph):
    '''
    input_batch: shape = [-1, height(240), width(320), 1]
    model_setting:
        training_layer_init_mode
        activation_mode
    '''
    # 1. init
    init_mode = model_setting['training_layer_init_mode']
    activation_mode = model_setting['activation_mode']
    
    dropout_prob = model_setting['dropout_prob']

    image_resolution_str = model_setting['image_resolution'].split(' ')
    image_resolution = [int(i) for i in image_resolution_str]
    input_height = image_resolution[0]
    input_width = image_resolution[1]
    input_channel = [8,16,32,64]
    expension_factor = 6
    stage = [1,2,2,1] # each stage includes N block

    if model_setting['color_mode'] == 'rgb':
        input_channel.insert(0,3)
    elif model_setting['color_mode'] == 'gray':
        input_channel.insert(0,1)

    # 2.0
    c0 = conv_layer(input_batch, 
                    [3, 3, input_channel[0], input_channel[1]],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='input_layer_c0')
    c0 = tf.layers.BatchNormalization(momentum=0.9,scale=True)(c0)
    c0 = activation(c0, activation_mode)
    #di=dropout_with_mode(c0, dropout_prob_ph, activation_mode)
    di=c0

    # 2.1 residual
    for s in range(1, len(stage)+1):
        for i in range(1, stage[s-1]+1):
            bi = mobilenetv2_block_1(di, [3,3,input_channel[s],input_channel[s]],
                                    expension_factor=expension_factor,
                                    name='stage%d_block1_%d'%(s,i))

            #di = dropout_with_mode(bi,dropout_prob_ph, activation_mode)
            di = bi

        if s < len(stage):
            di = mobilenetv2_block_2(di, [3,3,input_channel[s],input_channel[s+1]],
                                     expension_factor=expension_factor,
                                     name='stage%d_block2'%(s))

    # 3. global avg_pool
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(di)

    # 4. fc
    logits = fc_layer(avg_pool, input_channel[-1], 2, mode=init_mode, name = 'output_fc')
    softmax_prob = tf.keras.activations.softmax(logits)
    #logits = tf.reshape(logits, [-1,])
    return logits, softmax_prob, dropout_prob_ph

def simple_convnet(input_batch, model_setting, dropout_prob_ph):
    # 1. init
    init_mode = model_setting['training_layer_init_mode']
    activation_mode = model_setting['activation_mode']
    
    dropout_prob = model_setting['dropout_prob']

    image_resolution_str = model_setting['image_resolution'].split(' ')
    image_resolution = [int(i) for i in image_resolution_str]
    input_height = image_resolution[0]
    input_width = image_resolution[1]
    if model_setting['color_mode'] == 'rgb':
        input_channel = 3
    elif model_setting['color_mode'] == 'gray':
        input_channel = 1

    # c
    c0 = conv_layer(input_batch, 
                    [5, 5, input_channel, 64],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='input_layer_c0')
    c0 = tf.keras.layers.BatchNormalization()(c0)
    c0 = activation(c0, activation_mode)
    d0=dropout_with_mode(c0, dropout_prob_ph, activation_mode)

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d0)

    c1 = conv_layer(max_pool, 
                    [3, 3, 64, 128],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='input_layer_c0')
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = activation(c1, activation_mode)
    d1=dropout_with_mode(c1, dropout_prob_ph, activation_mode)

    # 
    flatten = tf.keras.layers.Flatten()(d1)

    fc1 = fc_layer(flatten, 64*64*128, 256, init_mode, name = 'fc1')
    fc1 = activation(fc1, activation_mode)
    logits = fc_layer(fc1, 256, 2, init_mode, name = 'fc2')
    softmax_prob = tf.keras.activations.softmax(logits)
    
    return logits, softmax_prob, dropout_prob_ph

