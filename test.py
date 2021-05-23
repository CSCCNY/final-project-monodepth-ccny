import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/losses/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/models/')
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


if len(argv) > 3:
    dataset_path = argv[3]
    
    if argv[1] == 'unet128':
        model =  keras.models.load_model(argv[2], compile=False)
        data = nyu2_dataloader(dataset_path, 20, image_size=[128, 128, 3])
        X_test, y_test = data.get_nyu2_test_data(dataset_path, num_of_images = 10)
        preds = model.predict(X_test)
        for i in range(len(y_test)):
            prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
            pr = (np.reshape(prds1, newshape=(128, 128))*255)
            img_predicted = np.zeros((128, 128, 3))
            img_predicted[:,:,0] = pr
            img_predicted[:,:,1] = pr
            img_predicted[:,:,2] = pr
            
            plt.subplot(1,2,1)
            plt.imshow(np.array(pr, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
            plt.title("Predicted Depth")
            plt.axis('off')
            
            plt.subplot(1,2,2)
            plt.title("True")
            plt.axis('off')
            plt.imshow(np.array(y_test[i]*255, dtype=np.int16), cmap='magma')
            plt.savefig(str(argv[2])+'_{0}.png'.format(i), dpi=200, format='png')
        
    elif argv[1] == 'unet256':
        model =  keras.models.load_model(argv[2], compile=False)
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
            plt.savefig(str(argv[2])+'_{0}.png'.format(i), dpi=200, format='png')    
            
    elif argv[1] == 'res50':
        model =  keras.models.load_model(argv[2], compile=False)
        data = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        X_test, y_test = data.get_nyu2_test_data(dataset_path, num_of_images = 10)
        preds = model.predict(X_test)
        for i in range(len(y_test)):
            prds1 = np.reshape(preds[i], newshape=(preds[i].shape[0]*preds[i].shape[1]))
            pr = (np.reshape(prds1, newshape=(256, 256))*255)
            img_predicted = np.zeros((256, 256, 3))
            img_predicted[:,:,0] = pr
            img_predicted[:,:,1] = pr
            img_predicted[:,:,2] = pr
            
            plt.subplot(1,2,1)
            plt.imshow(np.array(pr, dtype=np.int16), cmap='magma') #, cmap ='CMRmap'
            plt.title("Predicted Depth")
            plt.axis('off')
            
            plt.subplot(1,2,2)
            plt.title("True")
            plt.axis('off')
            plt.imshow(np.array(y_test[i]*255, dtype=np.int16), cmap='magma')
            plt.savefig(str(argv[2])+'_{0}.png'.format(i), dpi=200, format='png')

else:
    print("Command received: ", argv[1])
    print("\nPlease define the model you want to test!\n")
    print("Command Example: python3 test.py unet128 unet128_128x128.hdf5 /absolute/path/to/dataset\n")
	
    # python3 test.py unet128 unet128_150ep_11.hdf5 /tmp/Projects2021/rgbd_dataset/nyu_data/





