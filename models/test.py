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
 
  
dataset_path = '/tmp/Projects2021/rgbd_dataset/Driveway2/170721_C0'
# dataset_path = '/tmp/Projects2021/rgbd_dataset/outdoor_test/test/HR'
# dataset_path = '/tmp/ORB_SLAM3/CCNY_SLAM/datasets/apt_002'
argv = sys.argv

#python3 test.py unet128

if argv[1] == 'unet128':
    model =  keras.models.load_model('best_model128.hdf5', compile=False)
    dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128])
    X_test, y_test = dtloader.get_testing_sample()
    preds = model.predict(X_test)
    for i in range(9):
        prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
        plt.subplot(2,1,1)
        pr = (np.reshape(prds1, newshape=(128, 128))*255)
        img_predicted = np.zeros((128, 128, 3))
        img_predicted[:,:,0] = pr
        img_predicted[:,:,1] = pr
        img_predicted[:,:,2] = pr
        plt.imshow(np.array(img_predicted, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
        plt.subplot(2,1,2)
        imgname = '/tmp/Projects2021/rgbd_dataset/Driveway2/170721_C0/depth/out_wb_00_170721_00020{0}_depth.png'.format(i+1)
        #imgname = dataset_path+str('/depth/000{0}.png'.format(i+1))
        depth_image = cv2.imread(imgname)
        depth_image = cv2.resize(depth_image, (128, 128))
        plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
        plt.savefig('res/ssmi_test_{0}.png'.format(i), dpi=200, format='png')
        cv2.imwrite('res/d{0}.png'.format(i+1), np.array(img_predicted, dtype=np.int16))
        # error_image = np.array(img_predicted, dtype=np.int16) - np.array(depth_image, dtype=np.int16)
        # cv2.imwrite('e{0}.png'.format(i+1), np.array(error_image, dtype=np.int16))

elif argv[1] == 'unet256':
    if argv[2] == 'indoor':
        dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test/'
        model =  keras.models.load_model('indoor_unet256.hdf5', compile=False)
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=[256, 256])
        X_test, y_test = dtloader.get_testing_sample()
        preds = model.predict(X_test)
        file_names = dtloader.depth_images
        for i, filename in enumerate(file_names):
            prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
            
            plt.subplot(2,1,1)
            pr = (np.reshape(prds1, newshape=(256, 256))*255)
            img_predicted = np.zeros((256, 256, 3))
            img_predicted[:,:,0] = pr
            img_predicted[:,:,1] = pr
            img_predicted[:,:,2] = pr
            plt.imshow(np.array(img_predicted, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
            plt.title("Predicted")
            
            plt.subplot(2,1,2)
            depth_image = cv2.imread(filename)
            depth_image = cv2.resize(depth_image, (256, 256))
            plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
            plt.title("True")
            plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
            plt.savefig('res_test_indoor/ssmi_u256_{0}.png'.format(i), dpi=200, format='png')
    else:
        model =  keras.models.load_model('best_model256.hdf5', compile=False)
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=[256, 256])
        X_test, y_test = dtloader.get_testing_sample()
        preds = model.predict(X_test)
        for i in range(9):
            prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
            plt.subplot(2,1,1)
            pr = (np.reshape(prds1, newshape=(256, 256))*255)
            img_predicted = np.zeros((256, 256, 3))
            img_predicted[:,:,0] = pr
            img_predicted[:,:,1] = pr
            img_predicted[:,:,2] = pr
            plt.imshow(np.array(img_predicted, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
            plt.subplot(2,1,2)
            imgname = '/tmp/Projects2021/rgbd_dataset/Driveway2/170721_C0/depth/out_wb_00_170721_00020{0}_depth.png'.format(i+1)
            #imgname = dataset_path+str('/depth/000{0}.png'.format(i+1))
            depth_image = cv2.imread(imgname)
            depth_image = cv2.resize(depth_image, (256, 256))
            plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
            plt.savefig('res/ssmi_test_{0}.png'.format(i), dpi=200, format='png')
            cv2.imwrite('res/d{0}.png'.format(i+1), np.array(img_predicted, dtype=np.int16))
    
elif argv[1] == 'res50':
    model =  keras.models.load_model('best_modelres50.hdf5')
    dtloader = dataloader_rgbd(dataset_path, 8, image_size=128)
    X_test, y_test = dtloader.get_testing_sample()
    preds = model.predict(X_test)
    prds1 = np.reshape(preds[0], newshape=(preds[0].shape[0]*preds[0].shape[1]))
    plt.subplot(2,1,1)
    pr = (np.reshape(prds1, newshape=(128, 128))*255)
    plt.imshow(np.array(pr*255, dtype=np.int16))
    plt.subplot(2,1,2)
    plt.imshow(np.array(y_test[0]*255, dtype=np.int16))
    plt.savefig('testres50.png', dpi=200, format='png')

else:
    print("Command received: ", argv[1])
    print("\nPlease define the model you want to test!\n")




