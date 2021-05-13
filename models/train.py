import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/losses/')
# from ssim_loss import *
from dataloaders import *
from unet128 import unet128
from unet256 import unet256
from res50 import res50
from res50_v2 import res50_v2
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
    loss1 = tf.reduce_mean(1-ssim)
    loss2 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    loss3 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return 0.7*loss1+loss2*0.15+0.15*loss3

if len(argv) > 1:
    if argv[1] == 'unet128':
        if argv[3] == 'indoor':
            dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor'
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128, 1])
        checkpoint = ModelCheckpoint('unet128_indoor.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = unet128(input_shape=[128, 128, 3])
        m.model.compile(optimizer='adam', loss=ssmi_loss1)
        m.model.summary()
        m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'unet256':
        if argv[3] == 'indoor':
            dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor'
            dtloader = dataloader_rgbd(dataset_path, 20, image_size=[256, 256, 1])
            checkpoint = ModelCheckpoint('unet256_indoor20.hdf5',
                                        monitor='loss',
                                        save_best_only=True)

            m = unet256(input_shape=[256, 256, 3])
            m.model.compile(optimizer='adam', loss=ssmi_loss1)
            m.model.summary()
            m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])
            
        elif argv[3] == 'indoor_fft':
            dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor'
            dtloader = dataloader_rgbdfft(dataset_path, 8, image_size=[256, 256, 1])
            checkpoint = ModelCheckpoint('indoor_fft_unet256.hdf5',
                                        monitor='loss',
                                        save_best_only=True)

            m = unet256(input_shape=[256, 256, 4])
            m.model.compile(optimizer='adam', loss=ssmi_loss1)
            m.model.summary()
            m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])
            
        else:
            dtloader = dataloader_rgbd(dataset_path, 20, image_size=[256, 256, 1])
            checkpoint = ModelCheckpoint('unet256_outdoor.hdf5',
                                        monitor='loss',
                                        save_best_only=True)

            m = unet256(input_shape=[256, 256, 3])
            m.model.compile(optimizer='adam', loss=ssmi_loss1)
            m.model.summary()
            m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'res50':
        # from tensorflow.keras.callbacks import TensorBoard
        # tensorboard = TensorBoard(log_dir='.\logs', histogram_freq=1, write_images=True)
        dataset_path = '/tmp/Projects2021/rgbd_dataset/nyu_data/'
        nyu2_dataset = nyu2_dataloader(dataset_path, 20, image_size=[128, 256, 3])
        # nyu2_val = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        # nyu2_val.val_setup(dataset_path)
        checkpoint = ModelCheckpoint('res50_nyu_128x256.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = res50_v2(input_shape=(128, 256, 3))
        m.model.compile(optimizer='adam', loss=ssmi_loss1)
        m.model.summary()
        m.model.fit(nyu2_dataset, epochs=int(argv[2]), callbacks=[checkpoint]) # ,validation_data=nyu2_val, tensorboard])
    else:
        print("\nPlease define the model you want to train and number of epochs!")
        print("Command Example: python3 train.py unet128 30\n")
else:
    print("\nPlease define the model you want to train and number of epochs!")
    print("Command Example: python3 train.py unet128 30\n")
