import numpy as np
import tensorflow as tf
from IPython import embed
import tf_slim

activations = {"relu": tf.nn.relu,
               "tanh": tf.nn.tanh,
               "sigmoid": tf.nn.sigmoid,
               "identity": tf.identity}

def _batch_normalization(x, phase):
    with tf.compat.v1.variable_scope("batch_norm"):
        return tf_slim.layers.batch_norm(x, 
                                            center=True, scale=True, 
                                            is_training=phase)

def _add_bias(x):
    with tf.compat.v1.variable_scope("bias"):
        return tf_slim.layers.bias_add(x)    

#def _activation(x, activation):
#    with tf.variable_scope("activation"):
#        return activation(x)

# ------------------------

def _batch_norm_and_activation(x, param, phase):
    if param["batch_norm"]:
        y = _batch_normalization(x, phase)
    else:
        y = _add_bias(x)
    if param["activation"]:
        y = activations[param["activation"]](y)
    return y
        
# -----------------------
        
def full_connection(x, param, phase, scope):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope("fc"):
            y = tf_slim.layers.fully_connected(x, param["num_outputs"],
                                                  activation_fn=None,
                                                  biases_initializer=None)
        y = _batch_norm_and_activation(y, param, phase)
    return y
    
def convolution2d(x, param, phase, scope):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope("conv"):
            y = tf_slim.layers.conv2d(x,
                                         param["num_outputs"],
                                         param["kernel"],
                                         param["stride"],
                                         padding='SAME',
                                         activation_fn=None,
                                         biases_initializer=None)
        y = _batch_norm_and_activation(y, param, phase)
    return y

def transposed_convolution2d(x, param, phase, scope):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope("deconv"):
            y = tf_slim.layers.conv2d_transpose(x,
                                                   param["num_outputs"],
                                                   param["kernel"],
                                                   param["stride"],
                                                   padding='SAME',
                                                   activation_fn=None,
                                                   biases_initializer=None)
        y = _batch_norm_and_activation(y, param, phase)
    return y

def reshape(x, param, phase, scope):
    shape = [-1] + param["shape"]
    print (shape)
    return tf.reshape(x, shape)
