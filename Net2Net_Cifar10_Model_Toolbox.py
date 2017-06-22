# Saypraseuth Mounsaveng
# Net2Net_Cifar10_Model_Toolbox.py v0 16.04.2017
#
# This project aims at testing the Net2Net functions described
# in Net2Net: ACCELERATING LEARNING VIA KNOWLEDGE TRANSFER
# Tianqi Chen , Ian Goodfellow, and Jonathon Shlens arvix 1511.05641
#
# Net2Net_Cifar10_Model_Toolbox.py contains functions to help design the 
# network used in the experiments:
#    init_weight
#    init_bias
#    conv2D
#    max_pooling_2x2
#
# ==================================================================
import tensorflow as tf

def init_weight(shape):
    """ Initialise a weights matrix of shape = shape using normal distribution
  
    Args:
      shape: Shape of the matrix to initialise. Numpy array of size unknown
  
    Returns:
      weight: Initialised weights matrix. Variable Tensor of dimension and unknown size
    """    
    weight = tf.truncated_normal(shape, stddev = 5e-2)
    return tf.Variable(weight, name='weight')

def init_bias(shape_bias):
    """ Initialise a bias vector of length = shape_bias to contant value 0.1
  
    Args:
      shape_bias: Length of the vector to initialise. Numpy array of dimension 1 and unknown size
  
    Returns:
      bias: Initialised bias vector. Variable Tensor of dimension 1 and size unknown.
    """    
    bias = tf.constant(0.1, shape=shape_bias)
    return tf.Variable(bias, name= 'bias')

def conv2D(x, W):
    """ Initialise a 2D Tensorflow convolutional layer based on input vector x and filter vector W
  
    Args:
      x: Input matrix of the convolutional layer. Tensorflow variable of unknown dimension 
      W: Input filter of the convolutional layer. 4D Tensorflow variable of dimension (filter_height, filter_width, nb_input_channels, nb_output_channels)
  
    Returns:
      : Initialised 2D convolutional layer. 2D Tensorflow convolutional layer with input x, filter W and stride (1,1)
    """  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = "SAME")

def max_pooling_2x2(x):
    """ Initialise a 2D Tensorflow convolutional layer based on input vector x and filter vector W
  
    Args:
      x: Input matrix of the max_pool layer. Tensorflow variable of unknown dimension 
  
    Returns:
      : Tensorflow max_pool layer with input x, input_window 2x2 and stride (2,2)
    """      
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")