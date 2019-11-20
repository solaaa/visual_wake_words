from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from utils import data_augment, group_conv_block, conv_layer, fc_layer, identity_block_v2,identity_block_v1, depthwise_seperable_conv, dropout_selu, conv_block, activation, dropout_with_mode, ds_identity_block


def create_model(input_batch, model_setting, model_architecture):
    if model_architecture == 'resnet_identity':
        dropout_prob_ph = tf.compat.v1.placeholder(tf.float32, name='dropout_prob')
        return resnet_identity(input_batch, model_setting, dropout_prob_ph)
    else: 
        raise Exception('model_architecture is not existing')

def resnet_identity(input_batch, model_setting, dropout_prob_ph):
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

    stage = [4,4,4] # each stage includes 2 block, max_pool layer is between stages
    channel = 32

    # 2.0
    c0 = conv_layer(input_batch, 
                    [7, 7, 1, channel],
                    [1,1,1,1],
                    mode=init_mode,
                    name='input_layer_c0')

    d0=dropout_with_mode(c0, dropout_prob_ph, activation_mode)
    di=d0

    # 2.1 residual
    for s in range(1, len(stage)+1):
        for i in range(1, stage[s-1]+1):
            bi = identity_block_v1(di, [3,3,channel,channel],[1,1,1,1],
                                   init_mode=init_mode, 
                                   activation_mode=activation_mode, 
                                   name='stage%d_block%d'%(s,i))

            di = dropout_with_mode(bi,dropout_prob_ph, activation_mode)

        if s < len(stage):
            # last stage won't be poolled by max_pool
            di = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(di)

    # 3. global avg_pool
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(di)

    # 4. fc
    logits = fc_layer(avg_pool, channel, 2, mode=init_mode, name = 'output_fc')
    softmax_prob = tf.keras.activations.softmax(logits)
    
    return logits, softmax_prob, dropout_prob_ph