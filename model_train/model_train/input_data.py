import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
import os
from PIL import Image

is_dct = 1


class DataProcessor(object):
    # 240p: 320×240
    # 360p: 480*360
    def __init__(self, data_path, batch_size, image_resolution, color_mode):
        self.train_path = os.path.join(data_path,'train2014')
        self.val_path = os.path.join(data_path,'val2014')
        self.batch_size = batch_size
        self.color_mode = color_mode

        self.label_path = os.path.join(data_path,'labels_thrd_0.005')
        self.train_labels = sio.loadmat(os.path.join(self.label_path, 'train_labels.mat'))
        self.val_labels = sio.loadmat(os.path.join(self.label_path, 'val_labels.mat'))
        image_resolution_str = image_resolution.split(' ')
        self.image_resolution = [int(i) for i in image_resolution_str]

        # dct matrix
        self.N = 8
        self.dct_matrix = np.zeros([self.N,self.N])
        for i in range(1,self.N+1):
            for j in range(1,self.N+1):
                if i==1:
                    self.dct_matrix[i-1][j-1] = 1/np.sqrt(self.N)
                else:
                    self.dct_matrix[i-1][j-1] = (1/np.sqrt(self.N))*np.sqrt(2)*np.cos(np.pi*(i-1)*(2*j-1)*0.5/self.N)
        self.dct_matrix_T = self.dct_matrix.T


    def image_process(self, img_path):
        # load image
        img = Image.open(img_path)
        # rgb to gray : mode={1，L，P，RGB，RGBA，CMYK，YCbCr，I，}
        if self.color_mode == 'rgb':
            img_gray = img.convert('RGB')
        elif self.color_mode == 'gray':
            img_gray = img.convert('L')
        elif self.color_mode == 'ycbcr':
            img_gray = img.convert('YCbCr')
        else:
            pass
        # resize, note that the Image.size is (width, height), while np.shape is (height, width)
        img_gray_resize = img_gray.resize((self.image_resolution[1], self.image_resolution[0]))
        # convert image obj to np (uint8 --> 0~1)
        img_data = np.array(img_gray_resize)
        #img_data = img_data/256
        if self.color_mode == 'rgb':
            img_data = img_data.reshape([self.image_resolution[0], self.image_resolution[1], 3])
        elif self.color_mode == 'gray':
            img_data = img_data.reshape([self.image_resolution[0], self.image_resolution[1], 1])
        elif self.color_mode == 'ycbcr':
            img_data = img_data.reshape([self.image_resolution[0], self.image_resolution[1], 3])
        else:
            pass
        return img_data

    def dct_process(self, img_data):
        '''
        reference: https://arxiv.org/pdf/2002.12416.pdf
        it is designed to use high-frequency info. to analyze and recognize targets by DCT.
        the input data is image in YCbCr form.
        the N of DCT window is 8, each window totally has 8*8*3(y,cb,cr) frequency components.
        according to reference paper, we choose: 6*6 left-up triangle in Y, 4*4 left-up triangle in Cb,Cr.(totally 41 componets)
        then for each window, we reshape to [1,1,41].
        e.g. input shape [128,128,3] ---> output shape [16,16,41]

        if needed, overlapping will be added.
        '''
        # currently, we only support YCbCr form image, with same W and H.
        assert img_data.shape[-1] == 3
        assert img_data.shape[0] == img_data.shape[1] # temp
        if img_data.shape[0]%self.N != 0:
            img_data = img_data[:-(img_data.shape[0]%self.N),:,:]
        if img_data.shape[1]%self.N != 0:
            img_data = img_data[:,:-(img_data.shape[1]%self.N),:]

        # we put all freq. in channel-axis, each 8*8 window obtain 1*1*41 freq. components.
        # test_2: obtain 1*1*27
        image_dct = np.zeros([img_data.shape[0]//self.N, img_data.shape[1]//self.N, 41])

        for k in range(3): # Y, Cb, Cr
            for i in range(img_data.shape[0]//self.N):
                for j in range(img_data.shape[1]//self.N):
                    # compute current window
                    cur = np.matmul(np.matmul(self.dct_matrix, 
                                              img_data[i*self.N:(i+1)*self.N, j*self.N:(j+1)*self.N, k:k+1].reshape([self.N, self.N])),
                                    self.dct_matrix_T)
                    # select
                    ## v1 21+10+10
                    #if k == 0: # Y
                    #    image_dct[i][j][:21] = np.concatenate([cur[0,:6], cur[1,:5], cur[2,:4], 
                    #                                           cur[3,:3], cur[4,:2], cur[5,:1]])
                    #elif k == 1: #Cb
                    #    image_dct[i][j][21:31] = np.concatenate([cur[0,:4], cur[1,:3], 
                    #                                             cur[2,:2], cur[3,:1]])
                    #else: # Cr
                    #    image_dct[i][j][31:41] = np.concatenate([cur[0,:4], cur[1,:3], 
                    #                                             cur[2,:2], cur[3,:1]])   
                    ## v2: 15+6+6                                                              
                    if k == 0: # Y
                        image_dct[i][j][:21] = np.concatenate([cur[0,:6], cur[1,:3], cur[2,:3], 
                                                               cur[3,:1], cur[4,:1], cur[5,:1]])
                    elif k == 1: #Cb
                        image_dct[i][j][21:31] = np.concatenate([cur[0,:2], cur[0,3:4], 
                                                                 cur[1,:1], cur[3,:1], cur[5,:1]])
                    else: # Cr
                        image_dct[i][j][31:41] = np.concatenate([cur[0,:2], cur[0,3:4], 
                                                                 cur[1,:1], cur[3,:1], cur[5,:1]])


        # test 1: quantize (x)
        #image_dct = np.round(image_dct)
        # test 2: normalize (freq.) channels (x) 
        #for c in range(41):
            #image_dct[:,:,c] = (image_dct[:,:,c] - np.mean(image_dct[:,:,c]))/np.std(image_dct[:,:,c])
        # test 3: normalize each image freq. domain
        image_dct = (image_dct - np.mean(image_dct))/np.std(image_dct)
        return image_dct


    def train_data_generator(self):
        file_list = os.listdir(self.train_path)
        #################
        # test small set
        #MAX_NUM = 3*64
        #file_list = file_list[:MAX_NUM]
        #################
        file_list_len = len(file_list)
        
        file_pt = 0
        label_keys = list(self.train_labels.keys())[3:] # index >=3 is valid
        while True:
            groundtruth_label_batch = []
            for i in range(self.batch_size):
                if file_pt == file_list_len:
                    file_pt = 0
                groundtruth_label_batch.append(self.train_labels[label_keys[file_pt]])
                img_path = os.path.join(self.train_path, file_list[file_pt])
                file_pt += 1
                img = self.image_process(img_path)
                if is_dct:
                    img_data = self.dct_process(img)
                else:
                    img_data = img
                if i == 0:
                    data_batch = img_data.reshape((1,) + img_data.shape)
                else:
                    data_batch = np.concatenate([data_batch, img_data.reshape((1,) + img_data.shape)],
                                                    axis=0)
            yield data_batch, np.array(groundtruth_label_batch).reshape([self.batch_size])


    def val_data_generator(self):
        file_list = os.listdir(self.val_path)
        file_pt = 0
        #################
        # test small set
        #MAX_NUM = 64*20
        #file_list = file_list[:MAX_NUM]
        #################
        label_keys = list(self.val_labels.keys())[3:] # index >=3 is valid
        while True:
            groundtruth_label_batch = []
            for i in range(self.batch_size):
                if file_pt == len(file_list):
                    file_pt = 0
                
                groundtruth_label_batch.append(self.val_labels[label_keys[file_pt]])
                img_path = os.path.join(self.val_path, file_list[file_pt])
                file_pt += 1
                img = self.image_process(img_path)
                if is_dct:
                    img_data = self.dct_process(img)
                else:
                    img_data = img
                if i == 0:
                    data_batch = img_data.reshape((1,) + img_data.shape)
                else:
                    data_batch = np.concatenate([data_batch, img_data.reshape((1,) + img_data.shape)],
                                                    axis=0)
            yield data_batch, np.array(groundtruth_label_batch).reshape([self.batch_size])


            
            
    
