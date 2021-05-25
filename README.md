# Depth Estimation for Mono Camera
This is the repository of the project for Deep Learning with TensorFlow 2.0 class.

![Alt Text](https://github.com/CSCCNY/final-project-monodepth-ccny/blob/main/results_unet128.gif)

Team members:

- [Ejup Hoxha](https://github.com/ehoxha91)

- [Stanislav Sotnikov](https://github.com/stansotn)

- [Santosh Suwal]( )

## Final Report ##

- [Final Report](report.md)

- [Presentation](https://drive.google.com/file/d/1nEn7QHxaVjn4Eyx47j6xY5RlOdzwV4QJ/view?usp=sharing) (via google drive link)

## Reading Materials ##

- [Original U-Net paper](https://arxiv.org/pdf/1505.04597.pdf) describes the basic architecture we use for 
  the depth prediction.
    
- [Monodepth 2](https://arxiv.org/pdf/1806.01260.pdf) A novel method of monocular depth estimation.
Desribes a self-supervised method, see occlusion loss.
  
- [D3V0](https://arxiv.org/pdf/2003.01060.pdf) Describes depth and pose estimation network.


## Deployment Instructions ##

Requires `python3.8`, `pip3`, `pipenv` to fetch dependencies automatically.

1. Clone repository and install dependencies.
```shell
git clone git@github.com:CSCCNY/final-project-monodepth-ccny.git
cd final-project-monodepth-ccny
pipenv install  # Installs dependencies
pipenv shell  # Enter virtual environment
```

#### Data Sources ####

All benchmarks we performed over the course of the project were trained on NYU v2 dataset.
Then evaluated on *NYU v2*, *Middlebury*, *DIML Indoor*, and *DIML Outdoor*. 
Only the dedicated **test** portion of *NYU v2*, *DIML Indoor*, and *DIML Outdoor* was used for evaluation.
In case with *Middlebury*, the entire dataset was used (less than 20 image - depth pairs).

 - [NYU v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) -- Default Indoor Set.
 - [Middlebury](https://vision.middlebury.edu/stereo/data/) -- Still Objects.
 - [DIML](https://dimlrgbd.github.io) -- Indoor and Outdoor.
