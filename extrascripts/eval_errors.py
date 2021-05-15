
import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
import tensorflow as tf
from dataloaders import *
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import backend as K
 

argv = sys.argv
dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
model =  keras.models.load_model('unet128_indoor.hdf5', compile=False)
dtloader = dataloader_rgbd(dataset_path, 38, image_size=[128, 128])
X_test, y_test = dtloader.get_testing_sample()
y_pred = model.predict(X_test)
y_pred = y_pred[:,:,:,0]*255
y_true = y_test*255

def mae(y_true, y_pred):
    error = y_pred-y_true
    return np.mean(np.abs(error))

def mse(y_true, y_pred):
    error = y_pred-y_true
    return np.mean(np.power(error,2))

def rmse(y_true, y_pred):
    error = y_pred-y_true
    return np.sqrt(np.mean(np.power(error,2)))

def log_rmse(y_true, y_pred):
    error = np.log(1+y_pred)-np.log(1+y_true)
    return np.sqrt(np.mean(np.power(error,2)))

print("MAE:     ", mae(y_true, y_pred))
# print("MSE:    ", mse(y_test, y_pred))
print("RMSE:    ", rmse(y_true, y_pred))
print("LogRMSE: ", log_rmse(y_true, y_pred))
