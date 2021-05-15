import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
from dataloaders import *
from keras.callbacks import ModelCheckpoint
import keras

# python3 train.py unet128

# dataset_path = '/tmp/Projects2021/rgbd_dataset/Driveway2/170721_C0'
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
    
    loss1 = tf.reduce_mean(1-ssim)/2.0
#    loss3 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    loss2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return 0.7*loss1+loss2*0.3 #+0.15*loss3

dataset_path = '/tmp/Projects2021/rgbd_dataset/nyu_data/'
nyu2_dataset = nyu2_dataloader(dataset_path, 15, image_size=[256, 256, 3])
checkpoint = ModelCheckpoint('res50_nyu_256x256.hdf5',
                            monitor='loss',
                            save_best_only=True)

model =  keras.models.load_model('res50_nyu_256x256.hdf5', compile=False)
#opt = Adam(0.0003)
model.compile(optimizer='adam', loss=ssmi_loss1)
model.summary()
model.fit(nyu2_dataset, epochs=int(argv[1]), callbacks=[checkpoint])
