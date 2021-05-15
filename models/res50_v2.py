import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, Concatenate, Dense, BatchNormalization, Dropout, MaxPool2D, Activation, Conv2DTranspose
from keras.regularizers import l2
import keras

class res50_v2(object):
    def __init__(self, input_shape=(256, 256, 3)):
        self.model = self.get_model(input_shape)

    def conv_block(self, input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, input, skip_features, num_filters):
        #x = UpSampling2D((2,2), interpolation='bilinear')(input)
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def get_model(self, input_shape):
        from keras.applications import ResNet50
        inputs = keras.layers.Input(input_shape)

        '''Load pre-trained resnet50 '''
        resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs) # no classifier layers needed.

        ''' Encoder '''
        s1 = resnet50.get_layer("input_1").output
        s2 = resnet50.get_layer("conv1_relu").output
        s3 = resnet50.get_layer("conv2_block3_out").output
        s4 = resnet50.get_layer("conv3_block4_out").output

        ''' Bottleneck '''
        bn = resnet50.get_layer("conv4_block6_out").output 

        ''' Decoder '''
        d1 = self.decoder_block(bn, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)                
        d4 = self.decoder_block(d3, s1, 64)

        outputs = Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid')(d4)
        mdl = keras.models.Model(inputs, outputs)

        return mdl
