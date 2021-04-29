import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/losses/')
from ssim_loss import *
from dataloaders import *
from unet128 import unet128
from unet256_v2 import unet256_v2
from res50 import res50
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt

dataset_path = '/tmp/Projects2021/rgbd_dataset/dataset1/'
argv = sys.argv

if len(argv) > 1:
    if argv[1] == 'unet128':
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=128)
        checkpoint = ModelCheckpoint('best_model128.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = unet128(input_shape=[128, 128, 3])
        m.model.compile(optimizer='adam', loss=ssim_loss())
        m.model.summary()
        m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'unet256':
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=256)
        checkpoint = ModelCheckpoint('best_model256.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = unet256_v2(input_shape=[256, 256, 3])
        m.model.compile(optimizer='adam', loss='mse')
        m.model.summary()
        m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])

    elif argv[1] == 'res50':
        dtloader = dataloader_rgbd(dataset_path, 8, image_size=128)
        checkpoint = ModelCheckpoint('best_modelres50.hdf5',
                                    monitor='loss',
                                    save_best_only=True)

        m = res50(input_shape=(128, 128, 3))
        m.model.compile(optimizer='adam', loss='mse')
        m.model.summary()
        m.model.fit(dtloader, epochs=int(argv[2]), callbacks=[checkpoint])
    else:
        print("\nPlease define the model you want to train and number of epochs!")
        print("Command Example: python3 train.py unet128 30\n")
else:
    print("\nPlease define the model you want to train and number of epochs!")
    print("Command Example: python3 train.py unet128 30\n")


    
