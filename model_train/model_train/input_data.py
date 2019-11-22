import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
import os
from PIL import Image

class DataProcessor(object):
    # 240p: 320×240
    # 360p: 480*360
    def __init__(self, data_path, batch_size, image_resolution):
        self.train_path = os.path.join(data_path,'train2014')
        self.val_path = os.path.join(data_path,'val2014')
        self.batch_size = batch_size
        
        self.label_path = os.path.join(data_path,'labels')
        self.train_labels = sio.loadmat(os.path.join(self.label_path, 'train_labels.mat'))
        self.val_labels = sio.loadmat(os.path.join(self.label_path, 'val_labels.mat'))
        image_resolution_str = image_resolution.split(' ')
        self.image_resolution = [int(i) for i in image_resolution_str]


    def image_process(self, img_path):
        # load image
        img = Image.open(img_path)
        # rgb to gray : mode={1，L，P，RGB，RGBA，CMYK，YCbCr，I，}
        img_gray = img.convert('L')
        # resize, note that the Image.size is (width, height), while np.shape is (height, width)
        img_gray_resize = img_gray.resize((self.image_resolution[1], self.image_resolution[0]))
        # convert image obj to np (uint8 --> 0~1)
        img_data = np.array(img_gray_resize)
        img_data = img_data/256
        img_data = img_data.reshape([self.image_resolution[0], self.image_resolution[1], 1])

        return img_data

    def train_data_generator(self):
        file_list = os.listdir(self.train_path)
        file_pt = 0
        label_keys = list(self.train_labels.keys())[3:] # index >=3 is valid
        while True:
            groundtrue_label_batch = []
            for i in range(self.batch_size):
                if file_pt == len(file_list):
                    file_pt = 0
                
                groundtrue_label_batch.append(self.train_labels[label_keys[file_pt]])
                img_path = os.path.join(self.train_path, file_list[file_pt])
                file_pt += 1
                img_data = self.image_process(img_path)
                if i == 0:
                    data_batch = img_data.reshape([1,self.image_resolution[0],self.image_resolution[1],1])
                else:
                    data_batch = np.concatenate([data_batch, 
                                                     img_data.reshape([1,self.image_resolution[0],self.image_resolution[1],1])],
                                                    axis=0)
            yield data_batch, np.array(groundtrue_label_batch).reshape([-1,])

    def val_data_generator(self):
        file_list = os.listdir(self.val_path)
        file_pt = 0
        label_keys = list(self.val_labels.keys())[3:] # index >=3 is valid
        while True:
            groundtrue_label_batch = []
            for i in range(self.batch_size):
                if file_pt == len(file_list):
                    file_pt = 0
                
                groundtrue_label_batch.append(self.val_labels[label_keys[file_pt]])
                img_path = os.path.join(self.val_path, file_list[file_pt])
                file_pt += 1
                img_data = self.image_process(img_path)
                if i == 0:
                    data_batch = img_data.reshape([1,self.image_resolution[0],self.image_resolution[1],1])
                else:
                    data_batch = np.concatenate([data_batch, 
                                                     img_data.reshape([1,self.image_resolution[0],self.image_resolution[1],1])],
                                                    axis=0)
            yield data_batch, np.array(groundtrue_label_batch).reshape([-1,])


            
            
    
