import os
import input_data
import scipy.io as sio
from pathlib import Path
#############################################
#
# to accelerate training, save the batch-data
#
#############################################

data_dir = r'E:\Visual Wake Words\data\coco_dataset'
training_data_dir = r'E:\Visual Wake Words\data\coco_dataset\train_resize\whole_data_batch_224'
val_data_dir = r'E:\Visual Wake Words\data\coco_dataset\val_resize\whole_data_batch_224'
if (Path(training_data_dir).exists()==False):
    os.mkdir(training_data_dir)
if (Path(val_data_dir).exists()==False):
    os.mkdir(val_data_dir)

data_processor = input_data.DataProcessor(data_dir, 64, '224 224', 'rgb')

training_generator = data_processor.train_data_generator()
val_generator = data_processor.val_data_generator()

TRAIN_NUM = 1290
VAL_NUM = 300

# train
for i in range(TRAIN_NUM):
    data_batch, data_labels = next(training_generator)
    sio.savemat(os.path.join(training_data_dir, 'batch_%d.mat'%(i)), 
                {'data':data_batch, 'label':data_labels})
# val
for i in range(VAL_NUM):
    data_batch, data_labels = next(val_generator)
    sio.savemat(os.path.join(val_data_dir, 'batch_%d.mat'%(i)), 
                {'data':data_batch, 'label':data_labels})