
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from plot_thread import plot_thread

import argparse
import os.path
import sys

import scipy.io as sio
import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

import time
import datetime
import platform
from pathlib import Path
import os
from utils import data_augment, get_mask_par, get_warp_par, get_current_time
import random
import matplotlib.pyplot as p

FLAGS = None

def main(_):
  
  global g_model_selected
  global g_model_path
  date_str = get_current_time()
  acc_list = []
  acc_list_moving_avg = [0.5]

  val_acc_list = []
  loss_list = []
  val_loss_list = []
  #dtype = '/zero2nine_fix'

  model_setting = {
      'training_layer_init_mode': FLAGS.training_layer_init_mode,
      'activation_mode': FLAGS.activation_mode,
      'image_resolution': FLAGS.image_resolution,
      'dropout_prob': FLAGS.dropout_prob,
      'batch_size': FLAGS.batch_size,
      'color_mode': FLAGS.color_mode
      }


  #MAX_TRAIN_BATCH = 2000
  #augment_rate = 0.2
  #is_augment = True


  #augment_rate = tf.compat.v1.placeholder(dtype=tf.float32,
  #                                        name='augment_rate')
  load_path_val = os.path.join(FLAGS.data_dir, 'val2014')
  val_file_list = os.listdir(load_path_val)
  val_data_size = len(val_file_list)

  # We want to see all the logging messages for this tutorial.
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.compat.v1.InteractiveSession()



  # get data generator
  #data_processor = input_data.DataProcessor(FLAGS.data_dir, FLAGS.batch_size, FLAGS.image_resolution, FLAGS.color_mode)
  #training_generator = data_processor.train_data_generator()
  #val_generator = data_processor.val_data_generator()

  # step and learning rate
  training_steps_list = [FLAGS.step_per_epoch]*FLAGS.epoch
  learning_rates_list = [FLAGS.start_learning_rate*(FLAGS.learning_rate_decay**i) for i in range(0, FLAGS.epoch)]
  print(learning_rates_list)

  #training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  #learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  #if len(training_steps_list) != len(learning_rates_list):
  #  raise Exception(
  #      '--how_many_training_steps and --learning_rate must be equal length '
  #      'lists, but are %d and %d long instead' % (len(training_steps_list),
  #                                                 len(learning_rates_list)))
  # 

  # tf gragh
  # input
  # ph = placeholder
  image_resolution_str = model_setting['image_resolution'].split(' ')
  image_resolution = [int(i) for i in image_resolution_str]
  image_height, image_width = image_resolution[0], image_resolution[1]

  if FLAGS.color_mode == 'gray':
      image_channel = 1
  else:
      image_channel = 3
  input_batch_ph = tf.compat.v1.placeholder(
      tf.float32, [None, image_height, image_width, image_channel], name='input_batch')


  # separate training stage and val stage for is_training
  selected_model = FLAGS.model_architecture
  logits, softmax_prob, dropout_prob_ph, model_detail= models.create_model(
      input_batch_ph,
      model_setting,
      selected_model)
  
  # Define loss and optimizer
  input_ground_truth_ph = tf.compat.v1.placeholder(
       #tf.int64, [FLAGS.batch_size,], name='groundtruth_input') #google version on github 08.2019
       tf.int64, [None,], name='input_groundtruth')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=input_ground_truth_ph, logits=logits)

  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    input_learning_rate_ph = tf.compat.v1.placeholder(
        tf.float32, [], name='input_learning_rate')
    if FLAGS.optimizor == 'Adam':
        train_step = tf.compat.v1.train.AdamOptimizer(
            input_learning_rate_ph).minimize(cross_entropy_mean)
    elif FLAGS.optimizor == 'RMSprop':
        train_step = tf.train.RMSPropOptimizer(
            learning_rate=input_learning_rate_ph, momentum=FLAGS.momentum).minimize(cross_entropy_mean)
    else:
        raise Exception("plz set optimizor!")

  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, input_ground_truth_ph)

  #confusion_matrix = tf.math.confusion_matrix(labels=input_ground_truth_ph, 
  #                                            predictions=predicted_indices,
  #                                            num_classes=2)
# -----------------------------------------------------------------------------------------------------
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.compat.v1.summary.scalar('accuracy', evaluation_step)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)

  saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.compat.v1.summary.merge_all()
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/validation')
  tf.compat.v1.global_variables_initializer().run()
  start_step = 1

  tf.compat.v1.logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.io.write_graph(sess.graph_def, FLAGS.train_dir,
                       g_model_selected + '.pbtxt')

  # -----------------------------------------------------------------------------------------------------
  #################################################
  # load check point
  # 2019_12_23_13_085 : 85% ACC, 128*128 
  #################################################
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())    
  saver.restore(sess, r'E:\Visual Wake Words\script\model_train\model_train\models\resnet\2019_12_23_13_085\speech_commands_train\resnet.ckpt-64500')           

  #################################################


  ####################################
  # restore model and training details
  ####################################
  detail_path = '.\\models\\'+g_model_selected+'\\'+date_str+'\\'+'training_details.txt'
  with open(detail_path, 'w') as f:
      f.write('model: %s \n'%(g_model_selected))
      f.write('-----------------model setting-------------------' +'\n')
      f.write('input_channel: '+str(model_detail['input_channel']) +'\n')
      f.write('expension_factor: '+str(model_detail['expension_factor']) +'\n')
      f.write('stage: '+str(model_detail['stage']) +'\n')
      f.write('-----------------training setting-------------------' +'\n')
      f.write('start_learning_rate: %f \n'%(FLAGS.start_learning_rate))
      f.write('learning_rate_decay: %f \n'%(FLAGS.learning_rate_decay))
      f.write('dropout_prob: %f \n'%(FLAGS.dropout_prob))
      f.write('epoch: %d \n'%(FLAGS.epoch))
      f.write('step_per_epoch: %d \n'%(FLAGS.step_per_epoch))
      f.write('batch_size: %d \n'%(FLAGS.batch_size))
      f.write('optimizor: %s \n'%(FLAGS.optimizor))

  ####################################



  print('Training started...')
  #starting time
  t0=time.time()

  # Training loop.
  training_steps_max = np.sum(training_steps_list)

  is_plot_acc=False
  if is_plot_acc:
      plot_thr = plot_thread()
      plot_thr.daemon = True
      plot_thr.start()
  print('======================================..')
  for training_step in range(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break

  # Pull training data and labels that we'll use for training.
    #data_batch, data_labels = next(training_generator)
    
    training_data = sio.loadmat(os.path.join(FLAGS.data_dir,'train_resize','whole_data_batch_224',
                                             'batch_%d.mat'%(training_step%FLAGS.step_per_epoch)))
    data_batch, data_labels = training_data['data'], training_data['label'][0]

    # change batch==64 to batch==FLAGS.batch_size(<64)
    index = random.sample(range(64), FLAGS.batch_size)
    data_batch, data_labels = data_batch[index],  data_labels[index]

    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _, logit_out = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step, logits
        ],
        feed_dict={ 
            input_batch_ph: data_batch,
            input_ground_truth_ph: data_labels,
            input_learning_rate_ph: learning_rate_value,
            dropout_prob_ph: FLAGS.dropout_prob
        })
    #####################
    #if training_step < 100000:
    #    print('---------------------------logit----------------------')
    #    print(logit_out[:5])
    #print('---------------------------ground_truth----------------------')
    #print(data_labels)
    #####################

    train_writer.add_summary(train_summary, training_step)
    tf.compat.v1.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)

    acc_list.append(train_accuracy)
    acc_list_moving_avg.append(0.95*acc_list_moving_avg[-1] + 0.05*train_accuracy)

    loss_list.append(cross_entropy_value)

    #print(np.array(acc_list).shape, np.array(acc_list_moving_avg[1:]).shape)

    if training_step%20 == 0 and is_plot_acc==True:
        plot_thr.set_param(acc_list, acc_list_moving_avg)


    # ---------validation-----------


    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      print('start validation...')
      total_accuracy = 0.
      total_loss = 0.

      #####################
      # test on small set
      #val_data_size = 64*20
      #####################

      #val_total_step = val_data_size//FLAGS.batch_size
      val_total_step = 300

      for i in range(0, val_total_step):
        # pull validation data and labels
        #val_data_batch, val_data_labels = next(val_generator)
        val_data = sio.loadmat(os.path.join(FLAGS.data_dir,'val_resize','whole_data_batch_224',
                                                 'batch_%d.mat'%(training_step%200)))
        val_data_batch, val_data_labels = val_data['data'], val_data['label'][0]
        # change batch==64 to batch==FLAGS.batch_size(<64)
        val_data_batch, val_data_labels = val_data_batch[index], val_data_labels[index]
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy, validation_cross_entropy_value = sess.run(
            [merged_summaries, evaluation_step, cross_entropy_mean],
            feed_dict={
                input_batch_ph: val_data_batch,
                input_ground_truth_ph: val_data_labels,
                dropout_prob_ph: 0.0
            })

        validation_writer.add_summary(validation_summary, training_step)

        total_accuracy += validation_accuracy
        total_loss += validation_cross_entropy_value

      total_accuracy = total_accuracy/val_total_step
      total_loss = total_loss/val_total_step

      val_acc_list.append(total_accuracy)

      val_loss_list.append(total_loss)


      tf.compat.v1.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, val_data_size))

    # Save the model checkpoint periodically.
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     g_model_selected + '.ckpt')
      tf.compat.v1.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)
    if FLAG_linux==0:
      my_log = {'loss': loss_list, 'val_loss': val_loss_list, 
            'acc': acc_list, 'val_acc': val_acc_list}
      sio.savemat('.\\models\\'+g_model_selected+'\\'+date_str+'\\'+'my_log.mat', my_log)

  t1=time.time()
  t_train=(t1-t0)/60.0
  #write my logfile
  mylogfile=g_model_path+'model_training_log.txt'
  mylogfile_ID=open(mylogfile, 'wt')
  mylogfile_ID.write('model used:'+g_model_selected+'\n')
  mylogfile_ID.write('Training started in:'+date_str+'\n')
  mylogfile_ID.write('Training time is: %.2f minutes\n' % t_train)


  mylogfile_ID.close()
  print('My log file generated.')
  if is_plot_acc:
    plot_thr.end_thread()
  if FLAG_linux==0:
      my_log = {'loss': loss_list, 'val_loss': val_loss_list, 
            'acc': acc_list, 'val_acc': val_acc_list}
      sio.savemat('.\\models\\'+g_model_selected+'\\'+date_str+'\\'+'my_log.mat', my_log)

if __name__ == '__main__':

  global FLAG_linux

  if platform.system()=='Linux':
    FLAG_linux=1
    timenow=datetime.datetime.now()
    date_str="d%s_t%02d%02d" %(str(timenow.date()), timenow.hour, timenow.minute)
  else:
    FLAG_linux=0
    date_str = get_current_time()




  g_model_selected='resnet'
  #g_model_selected='mobile_net_v2'
  #g_model_selected='devol_convnet'

  training_set_len = len(os.listdir(r'E:\Visual Wake Words\data\coco_dataset\train2014'))

  if (FLAG_linux==0):
    g_model_path='.\\models\\'+g_model_selected+'\\'+date_str+'\\' #windows version
  else:
    home_dir=home = str(Path.home())
    #g_model_path='/home/yli2000/research/trained_models/'+g_model_selected+'/'+date_str+'/' #linux version


  parser = argparse.ArgumentParser()
  if (FLAG_linux==0):
    parser.add_argument(
      '--data_dir',
      type=str,
      default=r'E:\Visual Wake Words\data\coco_dataset', #wys pc version
      #default='C:\\Training_data\\speech_dataset', #yli pc version
      #default='C:\Training_data\google_speech_dataset', #yli acer pc
      help="""\
      Where to download the training data to.
      """)
  else:
    parser.add_argument(
      '--data_dir',
      type=str,
      default='', #linux version
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      #default='/tmp/retrain_logs',
      #default='./models/conv_test/retrain_logs', # for pc test
      default=g_model_path+'retrain_logs',
      help='Where to save summary logs for TensorBoard.')

  parser.add_argument(
      '--train_dir',
      type=str,
      default=g_model_path+'speech_commands_train',
      help='Directory to write event logs and checkpoint.')

  parser.add_argument(
      '--save_step_interval',
      type=int,
      default = 500,
      help='Save model checkpoint every save_steps.')

  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default = 500,
      help='How often to evaluate the training results.')

  #####################
  # data augmentation #
  #####################
  parser.add_argument(
      '--rotation',
      type=float,
      default=30,
      help="""\
      max-degree of clockwise/counter-clockwise rotation.
      """)
    
  parser.add_argument(
      '--color_mode',
      type=str,
      default='rgb',
      help='rgb, gray')

  ####################
  # training settint #
  ####################

  parser.add_argument(
      '--image_resolution',
      type=str,
      default='224 224',
      help='240p(240 320), 360p(360 480) or other size.')

  parser.add_argument(
      '--training_layer_init_mode',
      type=str,
      default='tensorflow',
      help='tensorflow, keras, selu')

  parser.add_argument(
      '--activation_mode',
      type=str,
      default='relu6',
      help='relu, selu, chip_relu, leaky_relu, relu6')

  parser.add_argument(
      '--dropout_prob',
      type=float,
      default=0.05,
      help='0~1')

  parser.add_argument(
      '--model_architecture', 
      type=str,
      default=g_model_selected,
      help='What model architecture to use')

  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='How many items to train with at once',)

  parser.add_argument(
      '--epoch',
      type=int,
      default=3,
      help='How many training epochs to run',)

  parser.add_argument(
      '--step_per_epoch',
      type=int,
      #default=500,
      default=1290,
      help='How many training step per epoch',)

  #########################
  # learning rate setting #
  #########################
  parser.add_argument(
      '--start_learning_rate',
      type=float,
      default=0.0005,
      #default=0.045,
      help='learning_rate will decay by epoch',)
  parser.add_argument(
      '--learning_rate_decay',
      type=float,
      default=0.95,
      help='learning_rate decay',)
  #####################
  # optimizor setting #
  #####################
  parser.add_argument(
      '--optimizor',
      type=str,
      default='Adam',
      help='Adam, RMSprop',)
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='optimizor momentum',)




  FLAGS, unparsed = parser.parse_known_args()

  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  #main()


