# Variational Recurrent Neural Network

Variational Recurrent Neural Network implemented using TensorFlow.

## Requirements
This code was tested using :
- Python 3.8.5
- Tensorflow 2.4.1
- NumPy 1.19.5

## Implementation
- The model used in Denton and Fergus, "Stochastic Video Generation with a Learned Prior" [paper website](https://sites.google.com/view/svglp/)

## Example
```
$ cd train
$ python ../src/learn.py
  (It may take several minutes to download moving MNIST dataset for the first time)
$ python ../src/generate.py
```
