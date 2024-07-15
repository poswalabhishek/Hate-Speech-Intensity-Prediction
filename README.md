# Research

This repository aims to build up on previous research conducted in normalizing hatespeech. In order to do this, we will be using a dataset that has been previously used in the same context. The dataset can be found [here](https://github.com/poswalabhishek/NLP-Research/tree/main/data) 

To achieve the normalizing effect, we have divided the task in 3 subsets as follows:

1. Intensity Detection
2. Span Detection
3. Text Normalization/Rephrasal

## Enviornment Setup for windows-native:

#### Create a conda enviornment
conda create --name research python=3.9 
conda activate research 

#### Install compatible Tensorflow with cudatoolkit*
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow<2.11" 
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#### Install compatible version of Pytorch with compatible toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -c "import torch; print(torch.cuda.is_available())"

#### Install other dependencies from requirements.txt
pip install -r requirements.txt 

### Make the kernel visible in jupyter notebook
pip install ipykernel
python -m ipykernel install --user --name=research