import tensorflow as tf
import numpy as np
import math
import cv2
import os
import keras

class dataloader_rgbd(tf.keras.utils.Sequence): #
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
        self.image_size = image_size
        self.min_depth = 999999
        self.max_depth = 0
        for depthimg in self.depth_images:
            depth = cv2.imread(depthimg, -1)
            depth = cv2.resize(depth, (self.image_size, self.image_size))
            mx = np.max(depth)
            mn = np.min(depth)
            if mx > self.max_depth:
                self.max_depth = mx
            if mn < self.min_depth:
                self.min_depth = mn
        self.on_epoch_end()

    def on_epoch_end(self):
        None
    
    def __len__(self):
        return len(self.rgb_images) // self.batch_size

    def __getitem__(self, index):
        rgb_batch = self.rgb_images[index * self.batch_size:(index + 1) * self.batch_size]
        depth_batch = self.depth_images[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(rgb_batch, depth_batch)
        return X, y

    def __get_data(self, rgb_imgs, depth_imgs):
        rgb_images=[]
        depth_images=[]
        for rgb_file in rgb_imgs:
            rbg_img = cv2.imread(rgb_file) #, cv2.IMREAD_COLOR)
            rgb_img = cv2.resize(rbg_img, (self.image_size, self.image_size))
            rgb_img = rgb_img/float(255.0)
            rgb_images.append(rgb_img)
            
        for depth_file in depth_imgs:
            depth_image = cv2.imread(depth_file, -1)
            depth_image = cv2.resize(depth_image, (self.image_size, self.image_size))
            depth_image = depth_image/np.max(depth_image) #(depth_image-self.min_depth)/(self.max_depth - self.min_depth)
            depth_images.append(depth_image)
            
        return np.array(rgb_images, dtype=np.float16), np.array(depth_images, dtype=np.float16)

    def get_testing_sample(self):
        return self.__get_data(self.rgb_images[0:10], self.depth_images[0:10])
    
class DataGenerator(keras.utils.Sequence):
  
  def __init__(self, batch_size = 8, image_size = 128):
    self.batch_size = batch_size
    self.image_size = image_size

  def load(self, rgb_file, depth_file):
    image = cv2.imread(rgb_file)
    image = cv2.resize(image, (self.image_size, self.image_size)) # resize...
    depth = cv2.imread(depth_file, -1)
    depth = cv2.resize(depth, (self.image_size, self.image_size)) 

    image_normalized = image/np.max(image)
    depth_normalized = depth/np.max(depth)
    return image_normalized, depth_normalized

  def load_all(self, rgb_files, depth_files):
      images = []
      depths = []
      for i in range(len(rgb_files)):
          img, dpth = self.load(rgb_files[i], depth_files[i])
          images.append(img)
          depths.append(dpth)

      return np.array(images), np.array(depths)