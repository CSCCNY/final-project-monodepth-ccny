import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/losses/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/models/')
from dataloaders import *
from unet128 import unet128
from unet256_v2 import unet256_v2
from res50 import res50
from res50_v2 import res50_v2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt


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
    # ssim = tf.image.ssim_multiscale(y_true, y_pred, 
    #                        max_val=1.0, 
    #                        filter_size=11,
    #                        filter_sigma=1.5,
    #                        k1=0.01,
    #                        k2=0.03)
    
    loss1 = tf.reduce_mean(1-ssim) #/float(2.0)
    # loss2 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    loss3 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return 0.7*loss1+loss3*0.3 #+0.15*loss3

if len(argv) > 3:
    dataset_path = argv[3]
    if argv[1] == 'unet128':
        nyu2_dataset = nyu2_dataloader(dataset_path, 20, image_size=[128, 128, 3])
        #nyu2_val = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        #dtloader = dataloader_rgbd(dataset_path, 8, image_size=[128, 128, 1])
        checkpoint = ModelCheckpoint('unet128_128x128.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = unet128(input_shape=[128, 128, 3])
        m.model.compile(optimizer='adam', loss=ssmi_loss1)
        m.model.summary()
        m.model.fit(nyu2_dataset, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'unet256':
        nyu2_dataset = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        nyu2_val = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        nyu2_val.val_setup(dataset_path)
        checkpoint = ModelCheckpoint('unet256_256x256.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = unet256_v2(input_shape=[256, 256, 3])
        m.model.compile(optimizer='adam', loss=ssmi_loss1)
        m.model.summary()
        m.model.fit(nyu2_dataset, epochs=int(argv[2]), callbacks=[checkpoint])            

    elif argv[1] == 'res50':
        nyu2_dataset = nyu2_dataloader(dataset_path, 15, image_size=[256, 256, 3])
        nyu2_val = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        nyu2_val.val_setup(dataset_path)
        checkpoint = ModelCheckpoint('res50_nyu_256x256.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = res50_v2(input_shape=(256, 256, 3))
        m.model.compile(optimizer='adam', loss=ssmi_loss1)
        m.model.summary()
        m.model.fit(nyu2_dataset, validation_data=nyu2_val, epochs=int(argv[2]), callbacks=[checkpoint]) #, tensorboard])
    else:
        print("\nPlease define the model you want to train and number of epochs!")
        print("Command Example: python3 train.py unet128 30 /absolute/path/to/dataset\n")
else:
    print("\nPlease define the model you want to train, number of epochs and dataset path!")
    print("Command Example: python3 train.py unet128 30 /absolute/path/to/dataset\n")
