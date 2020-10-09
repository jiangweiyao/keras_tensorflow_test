# Quick Demonstration for Keras Tensorflow

Quick Demonstration to test if your Keras/Tensorflow is working. Also includes a short guide on saving and loading models generated using keras. Depends on Conda

## Clone Code from Repository
```
git clone https://github.com/jiangweiyao/keras_tensorflow_test.git
```

## Install Miniconda if Conda is not present

A script for installing and setting up Conda is included in this repository. Run it using the following command:
```
. ~/keras_tensorflow_test/install_miniconda.sh
```

Detailed instruction on the the Miniconda website if anything goes wrong:
https://conda.io/projects/conda/en/latest/user-guide/install/linux.html


## Build Conda Environment

#### For CPU Keras and Tensorflow
```
conda create -c conda-forge --name MachineLearning python=3 tensorflow=2.2.0
```

#### For GPU Keras and Tensorflow
```
conda create -c conda-forge --name MachineLearning-GPU python=3 tensorflow-gpu=2.2.0
```

## Activate Environment
#### GPU is demonstrated. 

```
conda activate MachineLearning-GPU
```

## Run test code
```
python3 ~/keras_tensorflow_test/kerastest.py
```

## Making and loading models
Models made in one script can be saved and used in another. The included ModelTrain.py and ModelPredict.py scripts show how to do this.

1. Go into the git repository
```
cd  ~/keras_tensorflow_test
```

2. Run the training script. The training script takes the input data (pima-indians-diabetes.data.csv) and trains a Neural Network using the data. Then, the trained model is saved as a h5 file named `model.h5`. An example model file (model.h5.bak) is included in this repository if you have trouble reproducing the environment. 
```
python ModelTrain.py
```

3. Load the load and predict. You should see that the same accuracy is generated from the prediction.
```
python ModelPredict.py
```

