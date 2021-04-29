import math
import numpy as np
import cv2

def ssim(y_true, y_pred):

    img1 = y_true
    img2 = y_pred
    # img1 = y_true.numpy()
    # img2 = y_pred.numpy()
    # img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
    # img2 = tf.convert_to_tensor(img2, dtype=tf.float32)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


depth_image1 = cv2.imread('/home/ehoxha/Projects2021/rgbd_dataset/dataset2/depth/d1.png', -1)
depth_image2 = cv2.imread('/home/ehoxha/Projects2021/rgbd_dataset/dataset2/depth/d2.png', -1)

# print(ssim(depth_image1, depth_image2))