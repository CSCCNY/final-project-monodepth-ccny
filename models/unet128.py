import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, Concatenate, Dense, BatchNormalization, Dropout, MaxPool2D, Activation
from keras.regularizers import l2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import keras

class unet128(object):
    def __init__(self, input_shape=[128, 128, 3]):
        self.model = self.UNet(in_shape = input_shape)

    def down_block(self, x, filters, kernel_s = (3,3), 
                padding_ = 'same', strides_ = 1, 
                activation_ = 'relu'):

        conv1_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_)(x)
        
        x = BatchNormalization()(conv1_layer)
        x = Activation(activation_)(x)
        
        conv2_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_)(x)
        
        x = BatchNormalization()(conv2_layer)
        x = Activation(activation_)(x)
        
        pooling_layer = MaxPool2D((2,2), strides=(2,2))(x)

        return conv2_layer, pooling_layer

    def up_block(self, x, skip, filters, kernel_s = (3,3), 
                        padding_ = 'same', strides_ = 1, 
                        activation_ = 'relu'):

        up_sampling_1 = UpSampling2D((2,2))(x)
        concat1 = Concatenate()([up_sampling_1, skip])

        conv1_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_)(concat1)
        
        x = BatchNormalization()(conv1_layer)
        x = Activation(activation_)(x)
        
        conv2_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_)(x)
        
        x = BatchNormalization()(conv2_layer)
        x = Activation(activation_)(x)

        return conv2_layer

    def bottleneck(self, x, filters, kernel_s = (3,3), 
                padding_ = 'same', strides_ = 1, 
                activation_ = 'relu'):

        conv1_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_)(x)
        
        x = BatchNormalization()(conv1_layer)
        x = Activation(activation_)(x)
        
        conv2_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_)(x)
        
        x = BatchNormalization()(conv2_layer)
        x = Activation(activation_)(x)
        
        return conv2_layer        

    def UNet(self, in_shape, filters = [16, 32, 64, 128, 256]):
        inputs = keras.layers.Input(in_shape)

        p0 = inputs
        c1, p1 = self.down_block(p0, filters[0])
        c2, p2 = self.down_block(p1, filters[1]) 
        c3, p3 = self.down_block(p2, filters[2]) 
        c4, p4 = self.down_block(p3, filters[3])

        bn = self.bottleneck(p4, filters[4])

        u1 = self.up_block(bn, c4, filters[3])
        u2 = self.up_block(u1, c3, filters[2])
        u3 = self.up_block(u2, c2, filters[1])
        u4 = self.up_block(u3, c1, filters[0])

        outputs = Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid')(u4)
        mdl = keras.models.Model(inputs, outputs)

        return mdl