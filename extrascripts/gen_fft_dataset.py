import cv2
import numpy as np
import os

dataset_path = '/home/ehoxha/Projects2021/rgbd_dataset/indoor_test'
rgb_images = os.listdir(str(str(dataset_path)+'/rgb/'))
rgb_images.sort()
rgb_images = [str(str(dataset_path)+'/rgb/') + file for file in rgb_images]

for rgbimage in rgb_images:
    im_gray = cv2.imread(rgbimage, cv2.IMREAD_GRAYSCALE)
    a = np.asarray(im_gray)
    b = np.real(np.fft.fft2(a))
    img_name = rgbimage.split('/')
    cv2.imwrite('fft_indoor/'+str(img_name[7]), np.array(b))
    print("Image: ", img_name[7])