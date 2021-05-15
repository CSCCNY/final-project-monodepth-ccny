import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
import tensorflow as tf
from dataloaders import *
from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D, Concatenate, Dense, BatchNormalization, Dropout, MaxPool2D
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
    

#dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
#dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128, 3])
#X_test, y_test = dtloader.get_testing_sample()
#model =  keras.models.load_model('res50_nyu_128x128.hdf5', compile=False)

dataset_path = '/tmp/Projects2021/rgbd_dataset/nyu_data/'
model =  keras.models.load_model('res50_nyu_256x256.hdf5', compile=False)
data = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
X_test, y_test = data.get_nyu2_test_data(dataset_path, num_of_images = 10)
preds = model.predict(X_test)
for i in range(len(preds)):
    prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
    pr = (np.reshape(prds1, newshape=(256, 256))*255)
    plt.imshow(np.array(pr, dtype=np.int16), cmap='magma')
    plt.axis("off")
    plt.savefig('res256x256/d_magma_{0}.png'.format(i), dpi=200, format='png')
    # img_predicted = np.zeros((128, 128, 3))
    # img_predicted[:,:,0] = pr
    # img_predicted[:,:,1] = pr
    # img_predicted[:,:,2] = pr
    cv2.imwrite('res256x256/d{0}.png'.format(i), np.array(pr, dtype=np.int16))                
