# Depth Estimation for Mono Camera
This is the repository of the project for Deep Learning with TensorFlow 2.0 class.

Team members:

- [Ejup Hoxha](https://github.com/ehoxha91)

- [Stanislav Sotnikov](https://github.com/stansotn)

- [Santosh Suwal]( )


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
