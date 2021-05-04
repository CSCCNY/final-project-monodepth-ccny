import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, Concatenate, Dense, BatchNormalization, Dropout, MaxPool2D, Activation
from keras.regularizers import l2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import keras

class res50(object):
    def __init__(self, input_shape=(128, 128, 3)):
        self.model = self.get_model(input_shape)

    def down_block(self, x, filters, kernel_s = (3,3), 
                padding_ = 'same', strides_ = 1, 
                activation_ = 'relu'):

        conv1_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
        
        x = BatchNormalization()(conv1_layer)
        x = Activation(activation_)(x)
        
        conv2_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
        
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
                            strides=strides_,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(concat1)
        
        x = BatchNormalization()(conv1_layer)
        x = Activation(activation_)(x)
        
        conv2_layer = Conv2D(filters, 
                            kernel_size=kernel_s, 
                            padding=padding_, 
                            strides=strides_,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
        
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

    def get_model(self, input_shape):
        from keras.applications import ResNet50
        inputs = keras.layers.Input(input_shape)

        '''Load pre-trained resnet50 '''
        resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs) # no classifier layers needed.

        ''' Encoder '''
        s1 = resnet50.get_layer("input_1").output           #128
        s2 = resnet50.get_layer("conv1_relu").output        #64, 64  filters
        s3 = resnet50.get_layer("conv2_block3_out").output  #32, 256 filters
        s4 = resnet50.get_layer("conv3_block4_out").output  #16, 512 filters

        ''' Bottleneck '''
        pooling_layer = MaxPool2D((2,2), strides=(2,2))(s4)
        bn = self.bottleneck(pooling_layer, filters = 1024)

        ''' Decoder '''
        d1 = self.up_block(bn, s4, 1024)
        d2 = self.up_block(d1, s3, 512)
        d3 = self.up_block(d2, s2, 256)
        d4 = self.up_block(d3, s1, 64)

        outputs = Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid')(d4)
        mdl = keras.models.Model(inputs, outputs)

        return mdl