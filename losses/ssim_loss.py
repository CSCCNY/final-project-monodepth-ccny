import tensorflow as tf
import keras

class ssim_loss(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="ssim_loss_func"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):        
        ssim = tf.image.ssim(y_true, y_pred, 
                              max_val=1.0, 
                              filter_size=11,
                              filter_sigma=1.5,
                              k1=0.01,
                              k2=0.03)
        return ssim