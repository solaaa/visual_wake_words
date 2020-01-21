
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio #this has to be kept 


model_dir = r'E:\Visual Wake Words\script\model_train\model_train\models\mobile_net_v2'
date = '2020_1_3_9'
check_point_dir = os.path.join(model_dir, date, 
                               'speech_commands_train', 
                               'mobile_net_v2.ckpt-13500.meta')
save_dir = os.path.join(model_dir, date, 'weights.mat')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(check_point_dir)
    saver.restore(sess,tf.train.latest_checkpoint(
                   os.path.join(model_dir, date,'speech_commands_train')))
    print('Graph:' + check_point_dir +'and checkpoint restored.')



