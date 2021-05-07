import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/losses/')
# from ssim_loss import *
from dataloaders import *
from unet128 import unet128
from unet256 import unet256
from res50 import res50
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# python3 train.py unet128

dataset_path = '/tmp/Projects2021/rgbd_dataset/Driveway2/170721_C0'
argv = sys.argv

def ssmi_loss1(y_true, y_pred):        
    y_true = tf.expand_dims(y_true, -1)
    y_pred = tf.expand_dims(y_pred, -1)
    ssim = tf.image.ssim(y_true, y_pred, 
                            max_val=1.0, 
                            filter_size=11,
                            filter_sigma=1.5,
                            k1=0.01,
                            k2=0.03)
    loss1 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss2 = tf.reduce_mean(1-ssim)
    return loss2*0.8+loss1*0.2

if len(argv) > 1:
    if argv[1] == 'unet128':
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128, 1])
        checkpoint = ModelCheckpoint('best_model128.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = unet128(input_shape=[128, 128, 3])
        m.model.compile(optimizer='adam', loss=ssmi_loss1)
        m.model.summary()
        m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'unet256':
        if argv[3] == 'indoor':
            dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor'
            # dtloader = dataloader_rgbd(dataset_path, 8, image_size=[256, 256, 1])
            dtloader = dataloader_rgbdfft(dataset_path, 8, image_size=[256, 256, 1])
            checkpoint = ModelCheckpoint('indoor_fft_unet256.hdf5',
                                        monitor='loss',
                                        save_best_only=True)

            m = unet256(input_shape=[256, 256, 4])
            m.model.compile(optimizer='adam', loss=ssmi_loss1)
            m.model.summary()
            m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])
        else:
            dtloader = dataloader_rgbd(dataset_path, 8, image_size=[256, 256, 1])
            checkpoint = ModelCheckpoint('best_model256.hdf5',
                                        monitor='loss',
                                        save_best_only=True)

            m = unet256(input_shape=[256, 256, 3])
            m.model.compile(optimizer='adam', loss=ssmi_loss1)
            m.model.summary()
            m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'res50':
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=128)
        checkpoint = ModelCheckpoint('best_modelres50.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = res50(input_shape=(128, 128, 3))
        opt = Adam(0.001)
        m.model.compile(optimizer=opt, loss='mse')
        m.model.summary()
        m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])
    else:
        print("\nPlease define the model you want to train and number of epochs!")
        print("Command Example: python3 train.py unet128 30\n")
else:
    print("\nPlease define the model you want to train and number of epochs!")
    print("Command Example: python3 train.py unet128 30\n")
