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
 
    
dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
model =  keras.models.load_model('unet128_indoor.hdf5', compile=False)
dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128])
X_test, y_test = dtloader.get_testing_sample()
preds = model.predict(X_test)
file_names = dtloader.depth_images
for i in range(len(preds)):
    prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
    pr = (np.reshape(prds1, newshape=(128, 128))*255)
    img_predicted = np.zeros((128, 128, 3))
    img_predicted[:,:,0] = pr
    img_predicted[:,:,1] = pr
    img_predicted[:,:,2] = pr
    cv2.imwrite('depth_pred/d{0}.png'.format(i+1), np.array(img_predicted, dtype=np.int16))
