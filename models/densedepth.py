"""
# Dense Depth #

This notebook is an attempt to recreate the [Dense Depth](https://arxiv.org/pdf/1606.00373.pdf) monocular depth estimation model.

The code and the dataset in the notebook is a modified and simplified version of the [original](https://github.com/ialhashim/DenseDepth) dense depth. Some differences from the original are:
 - Tensorflow model.
 - Simplified DenseNet encoder block.
 - No image augmentation prior to training.
 - Resolution of input is downscaled to 128 x 128 and respectively 64 x 64 output.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from keras.engine.topology import Layer, InputSpec
import keras.utils.conv_utils as conv_utils
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet169

import numpy as np
from io import BytesIO

"""## Model Architecture ##"""


class UpscaleBlock(Model):
    def __init__(self, filters, name):
        super(UpscaleBlock, self).__init__()
        self.up = UpSampling2D(size=(2, 2), interpolation="bilinear", name=name + "_upsampling2d")
        self.concat = Concatenate(name=name + "_concat")  # Skip connection
        self.convA = Conv2D(
            filters=filters, kernel_size=3, strides=1, padding="same", name=name + "_convA"
        )
        self.reluA = LeakyReLU(alpha=0.2)
        self.convB = Conv2D(
            filters=filters, kernel_size=3, strides=1, padding="same", name=name + "_convB"
        )
        self.reluB = LeakyReLU(alpha=0.2)

    def call(self, x):
        b = self.reluB(self.convB(self.reluA(self.convA(self.concat([self.up(x[0]), x[1]])))))
        return b


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.base_model = DenseNet169(
            input_shape=(None, None, 3), include_top=False, weights="imagenet"
        )
        print("Base model loaded {}".format(DenseNet169.__name__))

        # Create encoder model that produce final features along with multiple intermediate features
        outputs = [self.base_model.outputs[-1]]
        for name in ["pool1", "pool2_pool", "pool3_pool", "conv1/relu"]:
            outputs.append(self.base_model.get_layer(name).output)
        self.encoder = Model(inputs=self.base_model.inputs, outputs=outputs)

    def call(self, x: list):
        return self.encoder(x)


class Decoder(Model):
    def __init__(self, decode_filters: int):
        super(Decoder, self).__init__()
        self.conv2 = Conv2D(filters=decode_filters, kernel_size=1, padding="same", name="conv2")
        self.up1 = UpscaleBlock(filters=decode_filters // 2, name="up1")
        self.up2 = UpscaleBlock(filters=decode_filters // 4, name="up2")
        self.up3 = UpscaleBlock(filters=decode_filters // 8, name="up3")
        self.up4 = UpscaleBlock(filters=decode_filters // 16, name="up4")
        self.conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding="same", name="conv3")

    def call(self, features: list):
        x, pool1, pool2, pool3, conv1 = (
            features[0],
            features[1],
            features[2],
            features[3],
            features[4],
        )
        up0 = self.conv2(x)
        up1 = self.up1([up0, pool3])
        up2 = self.up2([up1, pool2])
        up3 = self.up3([up2, pool1])
        up4 = self.up4([up3, conv1])
        return self.conv3(up4)


class DepthEstimate(Model):
    def __init__(self):
        super(DepthEstimate, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(decode_filters=int(self.encoder.layers[-1].output[0].shape[-1] // 2))
        print("\nModel created.")

    def call(self, x):
        return self.decoder(self.encoder(x))


"""## Loss Function ##"""


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
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


"""## Data Loaders ##"""

from zipfile import ZipFile
from PIL import Image


# Path to nyu depth dataset root directory.
def extract_zip(input_zip: str):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def nyu_resize(img, resolution=128):
    from skimage.transform import resize

    return resize(
        img, (resolution, resolution), preserve_range=True, mode="reflect", anti_aliasing=True
    )
    # return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )


def get_nyu_data(batch_size, nyu_data_zipfile="/home/slava/downloads/nyu_data.zip"):
    data = extract_zip(nyu_data_zipfile)

    nyu2_train = list(
        (
            row.split(",")
            for row in (data["data/nyu2_train.csv"]).decode("utf-8").split("\n")
            if len(row) > 0
        )
    )
    nyu2_test = list(
        (
            row.split(",")
            for row in (data["data/nyu2_test.csv"]).decode("utf-8").split("\n")
            if len(row) > 0
        )
    )

    # shape_rgb = (batch_size, 480, 640, 3)
    # shape_depth = (batc
    shape_rgb = (batch_size, 128, 128, 3)
    shape_depth = (batch_size, 64, 64, 1)

    # Helpful for testing...
    if False:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth


def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    train_generator = NYU_BasicAugmentRGBSequence(
        data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth
    )
    test_generator = NYU_BasicRGBSequence(
        data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth
    )

    return train_generator, test_generator


class NYU_BasicAugmentRGBSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        data,
        dataset,
        batch_size,
        shape_rgb,
        shape_depth,
        is_flip=False,
        is_addnoise=False,
        is_erase=False,
    ):
        self.data = data
        self.dataset = dataset
        # self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
        #                             add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle

        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros(self.shape_rgb), np.zeros(self.shape_depth)

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N - 1)

            sample = self.dataset[index]

            x = np.clip(
                np.asarray(Image.open(BytesIO(self.data[sample[0]]))).reshape(480, 640, 3) / 255,
                0,
                1,
            )
            y = np.clip(
                np.asarray(Image.open(BytesIO(self.data[sample[1]]))).reshape(480, 640, 1)
                / 255
                * self.maxDepth,
                0,
                self.maxDepth,
            )
            y = self.maxDepth / y

            batch_x[i] = nyu_resize(x, 128)  # 480
            batch_y[i] = nyu_resize(y, 64)  #

            # if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

        return batch_x, batch_y


class NYU_BasicRGBSequence(tf.keras.utils.Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(self.shape_rgb), np.zeros(self.shape_depth)

        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N - 1)

            sample = self.dataset[index]

            x = np.clip(
                np.asarray(Image.open(BytesIO(self.data[sample[0]]))).reshape(480, 640, 3) / 255,
                0,
                1,
            )
            y = (
                np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32)
                .reshape(480, 640, 1)
                .copy()
                .astype(float)
                / 10.0
            )
            y = self.maxDepth / y

            print(x, y)

            batch_x[i] = nyu_resize(x, 128)
            batch_y[i] = nyu_resize(y, 64)

        return batch_x, batch_y


def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    train_generator = NYU_BasicAugmentRGBSequence(
        data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth
    )
    test_generator = NYU_BasicRGBSequence(
        data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth
    )

    return train_generator, test_generator


def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            '"channels_first", "channels_last". Received: ' + str(value)
        )
    return data_format


class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, "size")
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0], input_shape[1], height, width)
        elif self.data_format == "channels_last":
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == "channels_first":
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == "channels_last":
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None

        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {"size": self.size, "data_format": self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""## Train ##

To get cuda working:
 - Install latest [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
 - Install latest [Cudnn library](https://developer.nvidia.com/cudnn)
 - Specify path to libcudnn 
 ```shell
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
 ```
"""
if __name__ == "__main__":
    if "session" in locals() and session is not None:
        print("Close interactive session")
        session.close()

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = DepthEstimate()
    # model(tf.keras.Input(128, 128, 3))
    """Load nyu depth data. See definition of `get_nyu_data` to specify path."""

    # from google.colab import drive
    # drive.mount('/content/drive')

    train_generator, test_generator = get_nyu_train_test_data(32)

    """#### Callbacks ####
    
    1. Learning rate scheduler -- model doesn't gets stuck.
    2. Checkpoint -- save model every once in a while.
    """

    callbacks = []

    # Callback: Learning Rate Scheduler
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.7, patience=5, min_lr=0.00009, min_delta=1e-2
    )
    callbacks.append(lr_schedule)  # reduce learning rate when stuck

    # Callback: save checkpoints
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode="min",
            save_freq="epoch",
        )
    )

    optimizer = Adam(lr=0.0001, amsgrad=True)

    model.compile(loss=depth_loss_function, optimizer=optimizer)
    # model.load_weights("checkpoint/weights.10-0.12.hdf5")
    model.fit(
        train_generator,
        callbacks=callbacks,
        validation_data=test_generator,
        epochs=20,
        shuffle=True,
    )

    model.save("checkpoint/model.tf", save_format="tf")
