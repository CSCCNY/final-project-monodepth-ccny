import sys
sys.path.insert(1, '/tmp/Projects2021/depth_estimation/final-project-monodepth-ccny/dataloaders/')
from dataloaders import *
from models import *

dataset_path = '/tmp/Projects2021/rgbd_dataset/indoor_dataset/train_dataset'
dtloader = dataloarder_rgbd(dataset_path,10)

m = models(128)
m.unet_resnet50.compile(optimizer='adam', loss='mse')
m.unet_resnet50.summary()

m.unet_resnet50.fit_generator(dtloader, samples_per_epoch=50, nb_epoch=3)