import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/models/')
from dataloaders import *
from keras.callbacks import ModelCheckpoint
import keras
from keras.optimizers import Adam

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
    
    loss1 = tf.reduce_mean(1-ssim)/2
#    loss3 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    loss2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return 0.7*loss1+loss2*0.3 #+0.15*loss3

from random import randrange

if len(argv) > 5:
    dataset_path = argv[5]
    if argv[1] == "unet128":
        nyu2_dataset = nyu2_dataloader(dataset_path, 20, image_size=[128, 128, 3])
        checkpoint = ModelCheckpoint(argv[2]+"_"+str(randrange(0,100))+".hdf5",
                                    monitor='loss',
                                    save_best_only=True)

        model =  keras.models.load_model(argv[3]+".hdf5", compile=False)
        opt = Adam(0.003)
        model.compile(optimizer=opt, loss=ssmi_loss1)
        model.summary()
        model.fit(nyu2_dataset, epochs=int(argv[4]), callbacks=[checkpoint])
        
    if argv[1] == "unet256":
        nyu2_dataset = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        checkpoint = ModelCheckpoint(argv[2]+"_"+str(randrange(0,100))+".hdf5",
                                    monitor='loss',
                                    save_best_only=True)

        model =  keras.models.load_model(argv[3]+".hdf5", compile=False)
        model.compile(optimizer='adam', loss=ssmi_loss1)
        model.summary()
        model.fit(nyu2_dataset, epochs=int(argv[4]), callbacks=[checkpoint])
        
    elif argv[1] == "res50":
        nyu2_dataset = nyu2_dataloader(dataset_path, 20, image_size=[256, 256, 3])
        checkpoint = ModelCheckpoint(argv[2]+"_"+str(randrange(6,100))+".hdf5",
                                    monitor='loss',
                                    save_best_only=True)

        model =  keras.models.load_model(argv[3]+".hdf5", compile=False)
        opt = keras.optimizers.Adam(0.001)
        model.compile(optimizer=opt, loss=ssmi_loss1)
        model.summary()
        model.fit(nyu2_dataset, epochs=int(argv[4]), callbacks=[checkpoint])

else:
    print("Command received: ", argv)
    print("\nPlease define the model you want to continoue training!\n")
    print("Command Example: python3 resume_training.py res50 res_model_name old_model_name 10 /tmp/Projects2021/rgbd_dataset/nyu_data/n")

#python3 resume_training.py unet128 unet128_150ep unet128_128x128 50 /tmp/Projects2021/rgbd_dataset/nyu_data/
