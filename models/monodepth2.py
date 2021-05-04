import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, ReLU, MaxPooling2D
import keras
# Layers definitions:

class ReflectionPadding2D(keras.layers.Layer):
    # Defining a reflection pad layer for keras, copied completely from StackOverflow.
    # This is a mimic of pytorch ReflectionPad2d:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        return config
    
    
def res_block(inputs, layer, downsample=False):
    filters = inputs.shape[-1]
    filters *= 2 if downsample else 1
    strides = 2 if downsample else 1
    # ZeroPadding2D, 
    # Conv2D, 
    # BatchNormalization, 
    # ReLU
    pad1 = ZeroPadding2D(1)(inputs)
    name = 'en.layer' + str(layer[0]) + '.' + str(layer[1]) + '.'
    conv1 = Conv2D(filters, 3, 
                   activation='linear', 
                   use_bias=False, 
                   strides=strides,
                   name=name + 'conv1')(pad1)
    bn1 = BatchNormalization(momentum=0.9,
                             epsilon=1e-5,
                             name=name + 'bn1')(conv1)
    relu1 = ReLU()(bn1)
    
    # ZeroPadding2D, 
    # Conv2D, 
    # BatchNormalization, 
    # ~if upsampling: 
        # Skip Connection from the input layer
        # ReLu()
    # ~else:
        # Conv2D, 
        # BatchNormalization, 
        # ReLU
    pad2 = ZeroPadding2D(1)(relu1)
    conv2 = Conv2D(filters, 3, 
                   activation='linear',
                   use_bias=False,
                   name=name + 'conv2')(pad2)
    bn2 = BatchNormalization(momentum=0.9,
                             epsilon=1e-5,
                             name=name + 'bn2')(conv2)   
    
    # if it upsampling, add the skip layer
    if not downsample:
        add = bn2 + inputs
    else:
        name +='downsample.'
        conv3 = Conv2D(filters, 1, 
                       activation='linear',
                       use_bias=False, 
                       strides=2, 
                       name=name + '0')(inputs)
        bn3 = BatchNormalization(momentum=0.9, 
                                 epsilon=1e-5,
                                 name=name + '1')(conv3)
        add = bn2 + bn3
    relu2 = ReLU()(add)
    return relu2

def conv_block(size, inTensor, disp=False, cnt=''):
    # If disp=True, it is the last conv_block, filters=1
    name = 'dispconv' if disp else 'upconv'
    name = 'de.' + name + '.' + str(len("{0:b}".format(size)) - 5) + '.' + cnt
    filters = 1 if disp else size
    x = ReflectionPadding2D()(inTensor) # expand and reflect
    x = keras.layers.Conv2D(filters, 3, name=name)(x)
    # if not display, then use ELU, otherwise sigmoid (last layer)
    if not disp:
        x = keras.layers.ELU()(x)
    else:
        x = keras.activations.sigmoid(x)
    return x

def up_conv(size, firstTensor, secondTensor=None):
    # This block deals with upsampling and conv2d
    x = conv_block(size, firstTensor, cnt='0')
    x = keras.layers.UpSampling2D()(x)
    # if size of the images is bigger than 16
    # add skip layers with the secondTensor
    if size >16:
        x = keras.layers.concatenate([x, secondTensor], axis=-1)
    x = conv_block(size, x, cnt='1')
    
    
# Define the monodepth2 model.

inputs = keras.layers.Input(shape=(192,640,3))

encoder = []
decoder = []

# Encoder part:

# This is a mysterious thing... investigate monodepth2 paper
x = (inputs - 0.45) / 0.225 

x = ZeroPadding2D(3)(x) # add zeropadding to the images
x = Conv2D(64, 7, 
           strides=2,
           activation='linear',
           use_bias=False,
           name='conv1')(x)

x = BatchNormalization(momentum=0.9, 
                       epsilon=1e-5,
                       name='bn1')(x)
x = ReLU()(x)

# add this part to the encoder. (unique, that's why outside of the loop)
encoder.append(x) 
x = ZeroPadding2D(1)(x)
x = MaxPooling2D(3, 2)(x)
for i in range(1,5):
    x = res_block(x, (i, 0), i>1)  # (x - previous layer, i>1 == downsampling=False)
    x = res_block(x, (i, 1)     
    encoder.append(x)

# Decoder part

x= up_conv(256, encoder[4], encoder[3])
x = up_conv(128, x, encoder[2])
decoder.append(conv_block(128, x, disp=True))

x = up_conv(64, x, encoder[1])
decoder.append(conv_block(64, x, disp=True))

x = up_conv(32, x, encoder[0])
decoder.append(conv_block(32, x, disp=True))

x = up_conv(16, x)
decoder.append(conv_block(16, x, disp=True))

decoder = decoder[::-1]
model = keras.Model(inputs=inputs, outputs=outputs, name='depth')
model.summary()