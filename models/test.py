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
    if argv[2] == 'indoor':
        dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
        model =  keras.models.load_model('u128_indoor_mea_mse_ssim.hdf5', compile=False)
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128])
        X_test, y_test = dtloader.get_testing_sample()
        preds = model.predict(X_test)
        file_names = dtloader.depth_images
        for i, filename in enumerate(file_names):
            prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
            
            plt.subplot(1,2,1)
            pr = (np.reshape(prds1, newshape=(128, 128))*255)
            img_predicted = np.zeros((128, 128, 3))
            img_predicted[:,:,0] = pr
            img_predicted[:,:,1] = pr
            img_predicted[:,:,2] = pr
            plt.imshow(np.array(pr, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
            plt.title("Predicted Depth")
            
            plt.subplot(1,2,2)
            depth_image = cv2.imread(filename)
            depth_image = cv2.resize(depth_image[:,:,0], (128, 128))
            plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
            plt.savefig('res_mae_ssim/testu128_{0}.png'.format(i), dpi=200, format='png')
            # cv2.imwrite('res_10/d{0}.png'.format(i+1), np.array(img_predicted, dtype=np.int16))
    
    else:
            dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
            model =  keras.models.load_model('unet128_indoor.hdf5', compile=False)
            dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128])
            X_test, y_test = dtloader.get_testing_sample()
            preds = model.predict(X_test)
            file_names = dtloader.depth_images
            for i, filename in enumerate(file_names):
                prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
                
                plt.subplot(1,2,1)
                pr = (np.reshape(prds1, newshape=(128, 128))*255)
                img_predicted = np.zeros((128, 128, 3))
                img_predicted[:,:,0] = pr
                img_predicted[:,:,1] = pr
                img_predicted[:,:,2] = pr
                plt.imshow(np.array(pr, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
                plt.title("Predicted Depth")
                plt.axis('off')
                
                plt.subplot(1,2,2)
                depth_image = cv2.imread(filename)
                depth_image = cv2.resize(depth_image[:,:,0], (128, 128))
                plt.imshow(depth_image, cmap='magma')
                plt.title("Ground Truth")
                # plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
                plt.axis('off')
                plt.savefig('result_plots/plot_{0}.png'.format(i), dpi=200, format='png')
                # cv2.imwrite('pair_{0}.png'.format(i+1), out_img)

elif argv[1] == 'unet256':
    if argv[2] == 'indoor':
        dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
        model =  keras.models.load_model('unet256_indoor2   0.hdf5', compile=False)
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
            plt.savefig('res_unet256_indoor/testu256_{0}.png'.format(i), dpi=200, format='png')
    
    elif argv[2] == 'indoor_fft':
        dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_test'
        model =  keras.models.load_model('indoor_fft_unet256.hdf5', compile=False)
        dtloader = dataloader_rgbdfft(dataset_path, 8, image_size=[256, 256, 1])
        X_test, y_test = dtloader.__getitem__(0)
        preds = model.predict(X_test)
        for i in range(len(y_test)):
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
            depth_image = cv2.imread(dtloader.depth_images[i])
            depth_image = cv2.resize(depth_image, (256, 256))
            plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
            plt.title("True")
            plt.imshow(np.array(depth_image, dtype=np.int16), cmap='magma')
            plt.savefig('result_fftunet/test_{0}.png'.format(i), dpi=200, format='png')
            
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
    dataset_path = '/tmp/Projects2021/rgbd_dataset/nyu_data/'
    model =  keras.models.load_model('res50_nyu_128x256.hdf5', compile=False)
    data = nyu2_dataloader(dataset_path, 20, image_size=[256, 128, 3])
    X_test, y_test = data.get_nyu2_test_data(dataset_path, num_of_images = 10)
    preds = model.predict(X_test)
    for i in range(len(y_test)):
        prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
        pr = (np.reshape(prds1, newshape=(128, 256))*255)
#        img_predicted = np.zeros((128, 256, 3))
#        img_predicted[:,:,0] = pr
#        img_predicted[:,:,1] = pr
#        img_predicted[:,:,2] = pr
        
        plt.subplot(1,2,1)
        plt.imshow(np.array(pr, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
        plt.title("Predicted Depth")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.title("True")
        plt.axis('off')
        plt.imshow(np.array(y_test[i]*255, dtype=np.int16), cmap='magma')
        plt.savefig('resnet_res/plot_{0}.png'.format(i), dpi=200, format='png')
else:
    print("Command received: ", argv[1])
    print("\nPlease define the model you want to test!\n")




