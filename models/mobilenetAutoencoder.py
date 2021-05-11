import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, Activation, SeparableConv2D
import matplotlib.pyplot as plt
import tensorflow.keras as keras

class MobileNetAutoEncoder(object):
    def __init__(self, input_shape=[128, 128, 3]):
        self.model = self.get_model(input_shape = input_shape)

    def upBlock(self, x, skip, filters, kernel_s = (5,5), 
                        padding_ = 'same', strides_ = 1, 
                        activation_ = 'relu', name='upblock'):

        padding = 4 // 2
        conv1 = SeparableConv2D(filters, kernel_size=kernel_s,
                              padding='same', strides=strides_,
                              kernel_initializer='he_uniform',
                              activation=activation_, name='{}_conv'.format(name))(x)
        x = UpSampling2D((2,2), interpolation='nearest')(conv1)
        if skip is not None:
            x = x + skip
        return x      

    def get_model(self, input_shape):
        from tensorflow.keras.applications import MobileNet
        inputs = keras.layers.Input(input_shape)

        '''Load pre-trained MobileNet'''
        mobilenet = MobileNet(include_top=False, weights='imagenet', input_tensor=inputs)
        
        
        mobilenet_final_output_shape = mobilenet.layers[-1].output.shape
        decode_filters = int(mobilenet_final_output_shape[-1])
        
        '''Decoder'''
        decoder = Conv2D(
            filters=decode_filters, kernel_size=1, padding='same',
            input_shape=mobilenet_final_output_shape,
            name='decoder_conv_1')(mobilenet.output)

        decoder = self.upBlock(decoder, None, int(decode_filters / 2), name='uplock_1')
        decoder = self.upBlock(decoder, mobilenet.layers[30].output, int(decode_filters / 4), name='uplock_2')
        decoder = self.upBlock(decoder, mobilenet.layers[17].output, int(decode_filters / 8), name='uplock_3')
        decoder = self.upBlock(decoder, mobilenet.layers[7].output, int(decode_filters / 16), name='uplock_4')
        decoder = self.upBlock(decoder, None, int(decode_filters/32), name='upblock_5')

        outputs = Conv2D(1, kernel_size=(1,1), padding='same', name='decoder_final_conv')(decoder)
        outputs = Activation(activation='relu')(outputs)
        mdl = keras.models.Model(inputs, outputs)

        return mdl
    