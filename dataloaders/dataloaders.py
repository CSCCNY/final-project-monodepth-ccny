# rgb_path = '/tmp/Projects2021/rgbd_dataset/indoor_dataset/train_dataset/rgb/'


import tensorflow as tf
import numpy as np
import math
import cv2
import os

class dataloarder_rgbd(tf.keras.utils.Sequence): #
    def __init__(self, dataset_path, batch_size, image_size=128, shuffle=False):
        self.rgb_images = os.listdir(str(str(dataset_path)+'/rgb/'))
        self.rgb_images.sort()
        self.rgb_images = [str(str(dataset_path)+'/rgb/') + file for file in self.rgb_images]
        
        self.depth_images = os.listdir(str(str(dataset_path)+'/depth/'))
        self.depth_images.sort()
        self.depth_images = [str(str(dataset_path)+'/depth/') + file for file in self.depth_images]

        # Shuffle later
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_size = image_size

    def on_epoch_end(self):
        None
    
    def __len__(self):
        return len(self.rgb_images) // self.batch_size

    def __getitem__(self, index):
        rgb_batch = self.rgb_images[index * self.batch_size:(index + 1) * self.batch_size]
        depth_batch = self.depth_batch[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(rgb_batch, depth_batch)
        return X, y

    def __get_data(self, rgb_imgs, depth_imgs):
        rgb_images=[]
        depth_images=[]
        for rgb_file in rgb_imgs:
            rbg_img = cv2.imread(file_name) #, cv2.IMREAD_COLOR)
            rgb_img = cv2.resize(rbg_img, (self.image_size, self.image_size))
            rgb_image = rgb_image/np.max(rgb_image)
            rgb_images.append(rgb_img)
            
        for depth_file in depth_imgs:
            depth_image = cv2.imread(depth_file, -1)
            depth_image = cv2.resize(depth_image, (self.image_size, self.image_size))
            depth_image = depth_image/np.max(depth_image)
            depth_images.append(depth_image)
            
        return np.array(rgb_images), np.array(depth_image)
    
    
# dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_dataset/train_dataset'

# dtloader = dataloarder_rgbd(dataset_path,10)
# print(dtloader.rgb_images[10])
# print(dtloader.depth_images[10])