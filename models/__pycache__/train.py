from models import *

m = models(128)
m.unet_resnet50.compile(optimizer='adam', loss='mse')
m.unet_resnet50.summary()