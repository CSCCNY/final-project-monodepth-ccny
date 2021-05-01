import tensorflow as tf
import keras

class ssim_loss(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="ssim_loss_func"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):        
        y_pred1 = tf.squeeze(y_pred)
        ssim = tf.image.ssim(y_true, y_pred1, 
                              max_val=1.0, 
                              filter_size=11,
                              filter_sigma=1.5,
                              k1=0.01,
                              k2=0.03)
        print("\n\n")
        print(ssim)
        print("\n\n")
        return -ssim
    
    def depth_loss_function(self, y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
        # Point-wise depth
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = theta

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))