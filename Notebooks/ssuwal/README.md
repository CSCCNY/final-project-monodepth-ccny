# Notebook - Santosh Suwal

## File Structure

```cmd
FINAL-PROJECT-MONODEPTH-CCNY\NOTEBOOKS\SSUWAL
│   200_epoch_loss_resnet34.jpg
│   depth_estimation_using_UNET.ipynb
|   effb3-unet.png
|   EfficientNetB3-imagenet-with-nyu-dataset.ipynb
│   mobilenet-200-epochs.jpg
│   mobilenet-autoecoder.png
│   MobileNet-Autoencoder.ipynb
│   mobilenet-tensorboard-graph.png
│   MobileNet-with-nyu-dataset.ipynb
│   model.png
│   README.md
│   resnet34-tensorboard-graph.png
│   resnet34-unet-large-dataset.ipynb
│   resnet34-unet.ipynb
│   resnet34-unet.png
│   ResNet34-with-nyu-dataset.ipynb
│   resnet34unet.py
│   resnet35-unet-ssim-loss.png
│   ResNet35Unet_ssim_loss_func.ipynb
│   resnet50_encoder_unet.ipynb
│   resnet50_separableconv2d_unet.ipynb
│   Using-Segmentation-Model-Check.ipynb
├───depth
├───rgb

```

## File Description

1. 200_epoch_loss_resnet34.jpg

    Loss graph for train and validation on NYU V2 dataset for ResNet34

2. depth_estimation_using_unet

    This notebook contains base model UNET in tensorflow. It does train and test with available data with validation loss of 2% in 100 epochs.
    It also contains technique for filling holes in depth map generated by the model.

3. effb3-unet.png

    Graph of EfficientNet B3 AutoEncoder for depth estimation

4. EfficientNetB3-imagenet-with-nyu-dataset.ipynb

    Notebook containing EfficientNetB3

4. mobilenet-200-epochs.jpg

    Loss graph for train and validation on NYU V2 dataset for MobileNet

5. mobilenet-autoecoder.png

    Keras Model graph for MobileNet AutoEncoder

6. MobileNet-Autoencoder.ipynb

    Notebook with primary MobileNet AutoEncoder explored with test dataset

7. mobilenet-tensorboard-graph.png

    Tensorboard graph for mobilenet autoencoder.

8. MobileNet-with-nyu-dataset.ipynb

    MobileNet AutoEncoder trained with NYU V2 dataset with promising results.

9. model.png

    Figure of ResNet50 Unet implementation model graph.

10. resnet34-tensorboard-graph.png

    Tensorboard Graph of the ResNet34 Model

11. resnet34-unet-large-dataset.ipynb

    Unet implementation with ResNet34 as Encoder.Trained 50 epochs on large dataset with 1600+ images.
    [Dataset](https://dimlrgbd.github.io/#section-sampledata)

12. resnet34-unet.ipynb

    Unet implementation with ResNet34 as Encoder. No signigicant improvement over UNET.

13. resnet34-unet.png

    ResNet34 Unet model diagram.

14. ResNet34-with-nyu-dataset.ipynb

    ResNet34 trained on larger dataset of NYU V3 with good results, giving depth maps in test dataset.

15. resnet34unet.py

    ResNet34 AutoEncoder Model in python script.

16. resnet35-unet-ssim-loss.png

    ResNet34 AutoEncoder Model graph with Keras model plot.

17. ResNet35Unet_ssim_loss_func.ipynb

    Notebook testing ssim loss function with ResNet34

18. ResNet50_encoder_unet.pynb

    Unet implementation with ResNet50, with 50 epochs train without any improvements in result.

19. resnet50_separableconv2d_unet.ipynb

    ResNet50 Decoder using Separable Conv2D which is faster than Con2D and those pointwise and depthwise convolution.

20. Using-Segmentation-Model-Check.ipynb

    Using Segmentation Model Library to test different segmentation model on the data.