# Quick Test for Keras Tensorflow

Quick instructions to test if your Keras/Tensorflow is working. Depends on Conda

## Install Miniconda if Conda is not present

https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

## Build Conda Environment

#### For CPU Keras and Tensorflow
```
conda create -c conda-forge --name MachineLearning python=3.6.9 keras tensorflow
```

#### For GPU Keras and Tensorflow
```
conda create -c conda-forge --name MachineLearning-GPU python=3.6.9 keras-gpu tensorflow-gpu
```

## Activate Environment
#### GPU is demonstrated. 

```
conda activate MachineLearning-GPU
```

## Clone Code from Repository
```
git clone https://github.com/jiangweiyao/keras_tensorflow_test.git
```

## Run test code
```
python3 ~/keras_tensorflow_test/kerastest.py
```
