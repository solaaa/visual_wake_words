from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from utils import batch_norm,conv_block_v1,conv_block_v2, mobilenetv2_block_1, mobilenetv2_block_2, data_augment, group_conv_block, conv_layer, fc_layer, identity_block_v2,identity_block_v1, dropout_selu, activation, dropout_with_mode


def create_model(input_batch, model_setting, model_architecture):
    if model_architecture == 'resnet':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return resnet(input_batch, model_setting, dropout_prob_ph)
    elif model_architecture == 'freq_net':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return freq_net(input_batch, model_setting, dropout_prob_ph)
    elif model_architecture == 'mobile_net_v2':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return mobile_net_v2(input_batch, model_setting, dropout_prob_ph)
    elif model_architecture == 'devol_convnet':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return devol_convnet(input_batch, model_setting, dropout_prob_ph)
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

    input_channel = [16,32,64,128,256]
    kernel = 5
    stage = [0,0,0,0] # each stage includes N block
    mix_feature_channel = 512
    
    if model_setting['color_mode'] == 'rgb':
        input_channel.insert(0,3)
    elif model_setting['color_mode'] == 'gray':
        input_channel.insert(0,1)
   
    # 2.0
    c0 = conv_layer(input_batch, 
                    [kernel, kernel, input_channel[0], input_channel[1]],
                    [1,2,2,1],
                    mode=init_mode,
                    name='inp_c')
    c0 = batch_norm(c0,momentum=0.99,scale=True,trainable=True,name='inp_bn')
    c0 = activation(c0, activation_mode)
    c0 = dropout_with_mode(c0, dropout_prob_ph, activation_mode)
    di = c0

    # 2.1 residual
    for s in range(1, len(stage)+1):
        di = conv_block_v2(di, [kernel,kernel,input_channel[s],input_channel[s+1]], [1,2,2,1], 
                            is_training = True,
                            init_mode=init_mode, 
                            activation_mode=activation_mode,
                            name='stage%d_conv_block'%(s))
        di=dropout_with_mode(di, dropout_prob_ph, activation_mode)
        
        for i in range(1, stage[s-1]+1):
            di = identity_block_v1(di, [kernel,kernel,input_channel[s+1],input_channel[s+1]],[1,1,1,1],
                                   is_training = True,
                                   init_mode=init_mode, 
                                   activation_mode=activation_mode, 
                                   name='stage%d_id_block%d'%(s,i))
            di=dropout_with_mode(di, dropout_prob_ph, activation_mode)

    # 2.2 mix feature
    mix_feature = conv_layer(di, 
                    [1, 1, input_channel[-1], mix_feature_channel],
                    [1,1,1,1],
                    mode=init_mode,
                    name='mix_feature')
    mix_feature = batch_norm(mix_feature,momentum=0.99,scale=True,trainable=True,name='mix_feature_bn')
    mix_feature = activation(mix_feature, activation_mode)
    
    # 3. global avg_pool
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(mix_feature)
    avg_pool = dropout_with_mode(avg_pool, dropout_prob_ph, activation_mode)
    
    logits = fc_layer(avg_pool, mix_feature_channel, 2, mode=init_mode, name = 'output_fc')
    
    # 3.x 
    #flatten = tf.keras.layers.Flatten()(di)
    #fc1 = fc_layer(flatten, 24*24*32, 128, mode=init_mode, name = 'fc1')
    #fc1 = activation(fc1, activation_mode)
    #fc1 = dropout_with_mode(fc1, dropout_prob_ph, activation_mode)
    #logits = fc_layer(fc1, 128, 2, mode=init_mode, name = 'output_fc')

    softmax_prob = tf.keras.activations.softmax(logits)
    
    return logits, softmax_prob, dropout_prob_ph, {'input_channel': input_channel, 'expension_factor':1, 'stage':stage}

def freq_net(input_batch, model_setting, dropout_prob_ph):
    '''
    input_batch: shape = [-1, height(), width(), 41]
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
    input_height = image_resolution[0]//8
    input_width = image_resolution[1]//8

    input_channel = [32,64,128]

    kernel = 3
    stage = [1,1,1] # each stage includes N block
    mix_feature_channel = 256
    
    if model_setting['color_mode'] == 'rgb':
        input_channel.insert(0,3)
    elif model_setting['color_mode'] == 'gray':
        input_channel.insert(0,1)
    elif model_setting['color_mode'] == 'ycbcr':
        input_channel.insert(0,41)
    else:
        print('wrong color_mode !!!')
        pass
            
    ################################# v1 ##################################
    # 2.0
    # c0
    input_batch = batch_norm(input_batch,momentum=0.99,scale=True,trainable=True,name='bn_input')
    c0 = conv_layer(input_batch, 
                    [kernel, kernel, input_channel[0], input_channel[1]],
                    [1,1,1,1],
                    mode=init_mode,
                    name='c0')
    c0 = batch_norm(c0,momentum=0.99,scale=True,trainable=True,name='bn0')
    c0 = activation(c0, activation_mode)
    #c0 = dropout_with_mode(c0, dropout_prob_ph, activation_mode)
    d0 = c0

    ## c1 down sample [28,28] --> [14,14]
    #c1 = conv_layer(d0, 
    #                [kernel, kernel, input_channel[1], input_channel[2]],
    #                [1,2,2,1],
    #                mode=init_mode,
    #                name='c1')
    #c1 = batch_norm(c1,momentum=0.99,scale=True,trainable=True,name='bn1')
    #c1 = activation(c1, activation_mode)
    #c1 = dropout_with_mode(c1, dropout_prob_ph, activation_mode)
    #d1 = c1

    ## c2
    #c2 = conv_layer(d1, 
    #                [kernel, kernel, input_channel[2], input_channel[3]],
    #                [1,1,1,1],
    #                mode=init_mode,
    #                name='c2')
    #c2 = batch_norm(c2,momentum=0.99,scale=True,trainable=True,name='bn2')
    #c2 = activation(c2, activation_mode)
    #c2 = dropout_with_mode(c2, dropout_prob_ph, activation_mode)
    #d2 = c2

    ## c3 down sample [14,14] --> [7,7]
    #c3 = conv_layer(d2, 
    #                [kernel, kernel, input_channel[3], input_channel[4]],
    #                [1,2,2,1],
    #                mode=init_mode,
    #                name='c3')
    #c3 = batch_norm(c3,momentum=0.99,scale=True,trainable=True,name='bn3')
    #c3 = activation(c3, activation_mode)
    #c3 = dropout_with_mode(c3, dropout_prob_ph, activation_mode)
    #d3 = c3
    ################################# v1 ##################################


    
    ################################# v2 ##################################
    di = d0
    for s in range(len(stage)):
        for i in range(stage[s]):
            di = identity_block_v1(di, [kernel,kernel,input_channel[s+1],input_channel[s+1]],[1,1,1,1],
                                   is_training = True,
                                   init_mode=init_mode, 
                                   activation_mode=activation_mode, 
                                   name='stage%d_id_block%d'%(s,i))
            di=dropout_with_mode(di, dropout_prob_ph, activation_mode)
        if s != len(stage)-1:
            di = conv_block_v2(di, [kernel,kernel,input_channel[s+1],input_channel[s+2]], [1,2,2,1], 
                                is_training = True,
                                init_mode=init_mode, 
                                activation_mode=activation_mode,
                                name='stage%d_conv_block'%(s))
            di=dropout_with_mode(di, dropout_prob_ph, activation_mode)
    ################################# v2 ##################################


    # 2.1 mix feature
    mix_feature = conv_layer(di, 
                    [1, 1, input_channel[-1], mix_feature_channel],
                    [1,1,1,1],
                    mode=init_mode,
                    name='mix_feature')
    mix_feature = batch_norm(mix_feature,momentum=0.99,scale=True,trainable=True,name='mix_feature_bn')
    mix_feature = activation(mix_feature, activation_mode)
    
    # 3. global avg_pool
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(mix_feature)
    avg_pool = dropout_with_mode(avg_pool, dropout_prob_ph, activation_mode)
    
    logits = fc_layer(avg_pool, mix_feature_channel, 2, mode=init_mode, name = 'output_fc')
    
    # 3.x 

    softmax_prob = tf.keras.activations.softmax(logits)
    
    return logits, softmax_prob, dropout_prob_ph, {'input_channel': input_channel, 'expension_factor':1, 'stage':stage}


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
    input_channel = [16,32,64,128,256]
    expension_factor = [3,3,3,3]
    stage = [0,0,0,0] # each stage includes N block
    feature_mixed_channel = 256


    if model_setting['color_mode'] == 'rgb':
        input_channel.insert(0,3)
    elif model_setting['color_mode'] == 'gray':
        input_channel.insert(0,1)

    # 2.0
    c0 = conv_layer(input_batch, 
                    [3, 3, input_channel[0], input_channel[1]],
                    [1,2,2,1],
                    mode=init_mode,
                    padding='SAME',
                    name='input_layer_c0')
    c0 = tf.layers.BatchNormalization(momentum=0.99,scale=True, 
                                      trainable=True)(c0)
    c0 = activation(c0, activation_mode)
    #di=dropout_with_mode(c0, dropout_prob_ph, activation_mode)
    di=c0

    # 2 blocks
    for s in range(1, len(stage)+1):
        # None
        di = mobilenetv2_block_2(di, [5,5,input_channel[s],input_channel[s+1]],
                                     expension_factor=expension_factor[s-1],
                                     is_training = True,
                                     name='stage%d_block'%(s))
        #di = dropout_with_mode(di,dropout_prob_ph, activation_mode)
        for i in range(1, stage[s-1]+1):
            bi = mobilenetv2_block_1(di, [5,5,input_channel[s+1],input_channel[s+1]],
                                    expension_factor=expension_factor[s-1],
                                    is_training = True,
                                    name='stage%d_block_identity_%d'%(s,i))

            #di = dropout_with_mode(bi,dropout_prob_ph, activation_mode)
            di = bi

    # 3. feature-mixed conv. + global avg_pool
    feature_mixed = conv_layer(di, 
                    [1, 1, input_channel[-1], feature_mixed_channel],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='feature_mixed')
    feature_mixed = tf.layers.BatchNormalization(momentum=0.99,scale=True, 
                                      trainable=True)(feature_mixed)
    feature_mixed = activation(feature_mixed, activation_mode)
    
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(feature_mixed)
    avg_pool = dropout_with_mode(avg_pool, dropout_prob_ph, activation_mode)
    # 4. fc
    logits = fc_layer(avg_pool, feature_mixed_channel, 2, mode=init_mode, name = 'output_fc')
    softmax_prob = tf.keras.activations.softmax(logits)
    #logits = tf.reshape(logits, [-1,])
    return logits, softmax_prob, dropout_prob_ph, {'input_channel': input_channel, 'expension_factor':expension_factor, 'stage':stage}

def devol_convnet(input_batch, model_setting, dropout_prob_ph):
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

    # c0
    c0 = conv_layer(input_batch, 
                    [3, 3, input_channel, 32],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='input_layer_c0')
    c0 = tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(c0)
    c0 = activation(c0, activation_mode)
    d0=dropout_with_mode(c0, dropout_prob_ph, activation_mode)

    # c1
    max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d0)
    c1 = conv_layer(max_pool_1, 
                    [3, 3, 32, 16],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='c1')
    c1 = tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(c1)
    c1 = activation(c1, activation_mode)
    d1=dropout_with_mode(c1, dropout_prob_ph, activation_mode)

    # c2
    max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d1)
    c2 = conv_layer(max_pool_2, 
                    [3, 3, 16, 64],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='c2')
    c2 = tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(c2)
    c2 = activation(c2, activation_mode)
    d2=dropout_with_mode(c2, dropout_prob_ph, activation_mode)

    # c3
    #max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d1)
    c3 = conv_layer(d2, 
                    [3, 3, 64, 64],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='c3')
    c3 = tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(c3)
    c3 = activation(c3, activation_mode)
    d3=dropout_with_mode(c3, dropout_prob_ph, activation_mode)

    # c4
    max_pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d3)
    c4 = conv_layer(max_pool_4, 
                    [3, 3, 64, 32],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='c4')
    c4 =tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(c4)
    c4 = activation(c4, activation_mode)
    d4=dropout_with_mode(c4, dropout_prob_ph, activation_mode)

    # c5
    #max_pool_5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d4)
    c5 = conv_layer(d4, 
                    [3, 3, 32, 32],
                    [1,1,1,1],
                    mode=init_mode,
                    padding='SAME',
                    name='c5')
    c5 = tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(c5)
    c5 = activation(c5, activation_mode)
    d5=dropout_with_mode(c5, dropout_prob_ph, activation_mode)
    # 

    flatten = tf.keras.layers.Flatten()(d5)

    fc1 = fc_layer(flatten, 32*16*16, 128, init_mode, name = 'fc1')
    fc1 = tf.layers.BatchNormalization(momentum=0.99,scale=True,trainable=True)(fc1)
    fc1 = activation(fc1, activation_mode)
    fc1 = dropout_with_mode(fc1, dropout_prob_ph, activation_mode)

    logits = fc_layer(fc1, 128, 2, init_mode, name = 'fc2')
    softmax_prob = tf.keras.activations.softmax(logits)
    
    return logits, softmax_prob, dropout_prob_ph, {'input_channel': input_channel, 'expension_factor':expension_factor, 'stage':stage}

