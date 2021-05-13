import tensorflow as tf
import numpy as np
import math
import cv2
import os
import keras

class dataloader_rgbd(tf.keras.utils.Sequence): #
    def __init__(self, dataset_path, batch_size, image_size=[256, 128, 3], shuffle=False):
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
            rgb_img = cv2.resize(rbg_img, (self.image_size[0], self.image_size[1]))
            print(rgb_img.shape)
            rgb_img = rgb_img/float(255.0)
            rgb_images.append(rgb_img)
            
        for depth_file in depth_imgs:
            depth_image = cv2.imread(depth_file, -1)
            depth_image = cv2.resize(depth_image, (self.image_size[0], self.image_size[1]))
            d_mean = np.nanmean(depth_image)
            depth_image = np.where(depth_image == 0, d_mean, depth_image)
            depth_image = depth_image/float(255.0)  # this worked on outside data: 65535.0 #np.max(depth_image) #(depth_image-self.min_depth)/(self.max_depth - self.min_depth)
            depth_images.append(depth_image)
            
        return np.array(rgb_images, dtype=np.float16), np.array(depth_images, dtype=np.float16)

    def get_testing_sample(self):
        return self.__get_data(self.rgb_images[1100:1110], self.depth_images[1100:1110])
    
    
class dataloader_rgbdfft(tf.keras.utils.Sequence): #
    def __init__(self, dataset_path, batch_size, image_size=[256, 256], shuffle=False):
        self.rgb_images = os.listdir(str(str(dataset_path)+'/rgb/'))
        self.rgb_images.sort()
        self.rgb_images = [str(str(dataset_path)+'/rgb/') + file for file in self.rgb_images]
        
        self.fft_images = os.listdir(str(str(dataset_path)+'/fft/'))
        self.fft_images.sort()
        self.fft_images = [str(str(dataset_path)+'/fft/') + file for file in self.fft_images]
        
        self.depth_images = os.listdir(str(str(dataset_path)+'/depth/'))
        self.depth_images.sort()
        self.depth_images = [str(str(dataset_path)+'/depth/') + file for file in self.depth_images]
        
        # Shuffle later
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.on_epoch_end()

    def on_epoch_end(self):
        None
    
    def __len__(self):
        return len(self.rgb_images) // self.batch_size

    def __getitem__(self, index):               
        rgb_batch = self.rgb_images[index * self.batch_size:(index + 1) * self.batch_size]
        fft_batch = self.fft_images[index * self.batch_size:(index + 1) * self.batch_size]
        depth_batch = self.depth_images[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(rgb_batch, fft_batch, depth_batch)
        return X, y

    def __get_data(self, rgb_imgs, fft_imgs, depth_imgs):
        rgb_images=[]
        fft_images=[]
        depth_images=[]
        X_data = []
        for rgb_file in rgb_imgs:
            rbg_img = cv2.imread(rgb_file) #, cv2.IMREAD_COLOR)
            rgb_img = cv2.resize(rbg_img, (self.image_size[0], self.image_size[1]))
            rgb_img = rgb_img/float(255.0)
            rgb_images.append(rgb_img)
    
        for fft_file in fft_imgs:
            fft_img = cv2.imread(fft_file, -1)
            fft_img = cv2.resize(fft_img, (self.image_size[0], self.image_size[1]))
            fft_images.append(fft_img)
            
        for depth_file in depth_imgs:
            depth_image = cv2.imread(depth_file, -1)
            depth_image = cv2.resize(depth_image, (self.image_size[0], self.image_size[1]))
            d_mean = np.nanmean(depth_image)
            depth_image = np.where(depth_image == 0, d_mean, depth_image)
            depth_image = depth_image/float(255.0)  # this worked on outside data: 65535.0 #np.max(depth_image) #(depth_image-self.min_depth)/(self.max_depth - self.min_depth)
            depth_images.append(depth_image)
            
        # combine fft and rgb
        for i in range(len(rgb_imgs)):
            rgbimage = rgb_images[i]
            fftimage = fft_images[i]
            new_img = np.resize(rgbimage, (rgbimage.shape[0], rgbimage.shape[1], 4))
            new_img[:,:,3]=fftimage   
            X_data.append(new_img)                 
        return np.array(X_data, dtype=np.float16), np.array(depth_images, dtype=np.float16)

    def get_testing_sample(self):
        return self.__get_data(self.rgb_images[0:38], self.depth_images[0:38])
    
    
class DataGenerator(keras.utils.Sequence):
  
  def __init__(self, batch_size = 8, image_size = 128):
    self.batch_size = batch_size
    self.image_size = image_size

  def load(self, rgb_file, depth_file):
    image = cv2.imread(rgb_file)
    image = cv2.resize(image, (self.image_size, self.image_size))  # resize...
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


class nyu2_dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size, image_size=[256, 128, 3], shuffle=False):
        self.rgb_images, self.depth_images = self.read_csv_file(str(dataset_path)+'data/nyu2_train.csv')
        self.rgb_images = [str(dataset_path) + file for file in self.rgb_images]
        self.depth_images = [str(dataset_path) + file for file in self.depth_images]
        # Shuffle later
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.on_epoch_end()

    def read_csv_file(self, filename):
        file = open(filename)
        data = file.read()
        lines = data.split()
        rgb = []
        depth = []
        for line in lines:
            ln = line.split(',')
            rgb.append(ln[0])
            depth.append(ln[1])
        return rgb, depth

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
            rgb_img = cv2.resize(rbg_img, (self.image_size[0], self.image_size[1]))
            rgb_img = rgb_img/float(255.0)
            rgb_images.append(rgb_img)
            
        for depth_file in depth_imgs:
            depth_image = cv2.imread(depth_file, -1)
            depth_image = cv2.resize(depth_image, (self.image_size[0], self.image_size[1]))
            d_mean = np.nanmean(depth_image)
            depth_image = np.where(depth_image == 0, d_mean, depth_image)
            depth_image = depth_image/float(255.0)  # this worked on outside data: 65535.0 #np.max(depth_image) #(depth_image-self.min_depth)/(self.max_depth - self.min_depth)
            depth_images.append(depth_image)
            
        return np.array(rgb_images, dtype=np.float16), np.array(depth_images, dtype=np.float16)

    def get_nyu2_test_data(self, dataset_path, num_of_images = 10):
        rgbs, depths = self.read_csv_file(str(dataset_path)+'data/nyu2_test.csv')
        rgbs = [str(dataset_path) + file for file in rgbs]
        depths = [str(dataset_path) + file for file in depths]
        return self.__get_data(rgbs[0:num_of_images], depths[0:num_of_images])
    
    def val_setup(self, dataset_path, num_of_images = 400):
        self.rgb_images, self.depth_images = self.read_csv_file(str(dataset_path)+'data/nyu2_test.csv')
        self.rgb_images = [str(dataset_path) + file for file in self.rgb_images]
        self.depth_images = [str(dataset_path) + file for file in self.depth_images]
        self.rgb_images = self.rgb_images[:400]
        self.depth_images = self.depth_images[:400]
    
