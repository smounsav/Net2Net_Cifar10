# Saypraseuth Mounsaveng
# Net2Net.py v0.9 01.05.2017
#
# This project aims at testing the Net2Net functions described
# in Net2Net: ACCELERATING LEARNING VIA KNOWLEDGE TRANSFER
# Tianqi Chen , Ian Goodfellow, and Jonathon Shlens arvix 1511.05641
#
# Net2Net.py contains functions to extend a CNN:
#    net2WiderNet_CNN
#    net2WiderNet_FC
#    net2DeeperNet_FC
#    net2DeeperNet_CNN
#
# ==================================================================

import numpy as np

def net2WiderNet_CNN(W_U, b_U, W_UNext, size_ext, random_init=False,add_noise=False) :
    """ Implementation of net2WiderNet method for a CN layer
    Compute the new weights of layer U and U+1 (UNext) and the new biases of layer U 
    after having extended the number of output channels in layer U and the number
    of input channels in layer U+1 by size_ext

    Args:
      W_U: Weights matrix of layer U. 4D Numpy array of size 
          [filter_height_layer_U, filter_width_layer_U, old_nb_of_input_channels_layer_U, old_nb_of_output_channels_layer_U]
      b_U: Biases matrix of layer U. 1D Numpy array of size [old_nb_of_output_channels_layer_U]
      W_UNext: Weights matrix of layer U. 4D Numpy array of size 
          [filter_height_layer_UNext, filter_width_layer_UNext, old_nb_of_input_channels_layer_UNext, old_nb_of_output_channels_layer_UNext]
      size_ext: Number of new channels to add. Scalar.
      random_init=False: Flag to bypass the net2net extension method and use random init of weights instead. Boolean
      add_noise=False: Flag to add noise to the new weights initialised with net2net method. Boolean

    Returns:
      new_weights_matrix: New weights matrix of layer U. 4D Numpy array of size 
          [filter_height_layer_U, filter_width_layer_U, old_nb_of_input_channels_layer_U, old_nb_of_output_channels_layer_U + size_ext]
      new_biases_vector: New biases matrix of layer U. 1D Numpy array of size [old_nb_of_output_channels_layer_U + size_ext]
      new_weights_matrix_next_layerr: New weights matrix of layer UNext. 4D Numpy array of size 
          [filter_height_layer_UNext, filter_width_layer_UNext, old_nb_of_input_channels_layer_UNext + size_ext, old_nb_of_output_channels_layer_UNext]
    """

    new_weights_matrix = np.array(W_U)
    new_biases_vector = np.array(b_U)
    new_weights_matrix_next_layer = np.array(W_UNext)
	
    # Define the indexes matrix defining the output channels to be copied to the new output channels
    new_indexes = np.random.randint(new_weights_matrix.shape[3], size = size_ext)
    nb_occurences = np.bincount(new_indexes)

    for i in range(size_ext):
        # Extract output channel with index i from current output channels
        if random_init:
            new_output_channel = np.random.normal(0, scale=5e-2, size=(new_weights_matrix.shape[0], new_weights_matrix.shape[1], new_weights_matrix.shape[2]))
            new_biases_vector = np.append(new_biases_vector, 0.1)			
            new_input_channel = np.random.normal(0, scale=5e-2, size=(new_weights_matrix_next_layer.shape[0], new_weights_matrix_next_layer.shape[1], new_weights_matrix_next_layer.shape[3]))   			
        else:
    			# Extend weights matrix to add size_ext more output channels
            new_output_channel = np.array(new_weights_matrix[:, :, :, new_indexes[i]])
            # Copy existing bias of index new_indexes[i] to new bias with index len(current biases vector) + i
            new_biases_vector = np.append(new_biases_vector, new_biases_vector[new_indexes[i]])			
            # Extract input channel with index i from current input channels and divide weights by duplication factor
            new_input_channel = np.array((new_weights_matrix_next_layer[:, :, new_indexes[i], :]) * (1./(nb_occurences[new_indexes[i]] + 1)))            
    			# Add noise
            if add_noise:
                noise_out = np.random.normal(0, 0.01, new_output_channel.shape)
                new_output_channel += noise_out
                noise_in = np.random.normal(0, 0.01, new_input_channel.shape)
                new_input_channel += noise_in
		  # Add matrix dimensions to be able to concatenate new output channel to weigths matrix
        new_output_channel = new_output_channel[:, :, :, np.newaxis]
        new_weights_matrix = np.concatenate((new_weights_matrix, new_output_channel), axis=3)    
        # Add matrix dimensions to be able to concatenate new output channel to weigths matrix		
        new_input_channel = new_input_channel[:, :, np.newaxis, :]		
        new_weights_matrix_next_layer = np.concatenate((new_weights_matrix_next_layer, new_input_channel), axis=2)
    # Divide original layers by duplication factor
    for i in range(len(nb_occurences)):
        new_weights_matrix_next_layer[:, :, i, :] *= (1./(nb_occurences[i] + 1))
    return new_weights_matrix, new_biases_vector, new_weights_matrix_next_layer

def net2WiderNet_FC(W_U, b_U, W_UNext, size_ext, random_init=False, add_noise=False) :
    """ Implementation of net2WiderNet method for a FC layer
    Compute the new weights of layer U and U+1 (UNext) and the new biases of layer U 
    after having extended the number of output nodes in layer U and the number
    of input nodes in layer U+1 by size_ext

    Args:
      W_U: Weights matrix of layer U. 2D Numpy array of size [old_nb_of_input_nodes_layer_U, old_nb_of_output_nodes_layer_U]
      b_U: Biases matrix of layer U. 1D Numpy array of size [old_nb_of_output_nodes_layer_U]
      W_UNext: Weights matrix of layer U. 2D Numpy array of size [old_nb_of_input_nodes_layer_UNext, old_nb_of_output_nodes_layer_UNext]
      size_ext: Number of new nodes to add. Scalar.
      random_init=False: Flag to bypass the net2net extension method and use random init of weights instead. Boolean
      add_noise=False: Flag to add noise to the new weights initialised with net2net method. Boolean

    Returns:
      new_weights_matrix: New weights matrix of layer U. 2D Numpy array of size [old_nb_of_input_nodes_layer_U, old_nb_of_output_nodes_layer_U + size_ext]
      new_biases_vector: New biases matrix of layer U. 1D Numpy array of size [old_nb_of_output_nodes_layer_U + size_ext]
      new_weights_matrix_next_layer: New weights matrix of layer UNext. 4D Numpy array of size [old_nb_of_input_nodes_layer_UNext +size_ext, old_nb_of_output_nodes_layer_UNext]
    """
    new_weights_matrix = np.array(W_U)
    new_biases_vector = np.array(b_U)
    new_weights_matrix_next_layer = np.array(W_U)
    
    # Define the indexes matrix defining the output channels to be copied to the new output channels
    new_indexes = np.random.randint(new_weights_matrix.shape[1], size = size_ext)
    nb_occurences = np.bincount(new_indexes)   
    
    # Extend weights matrix to add size_ext more output channels
    for i in range(size_ext):
        if random_init:
            new_output_channel = np.random.normal(0, 5e-2, new_weights_matrix.shape[1])     
            new_biases_vector = np.append(new_biases_vector, 0.1)
            new_input_channel = np.random.normal(0, 5e-2, size=(new_weights_matrix_next_layer.shape[1]))               
        else:
            # Extract output channel with index i from current output channels
            new_output_channel = new_weights_matrix[:, new_indexes[i]]
            # Copy existing bias of index new_indexes[i] to new bias with index len(current biases vector) + i
            new_biases_vector = np.append(new_biases_vector, new_biases_vector[new_indexes[i]])            
            # Extract output channel with index i from current output channels and divide weights by duplication factor
            new_input_channel = (new_weights_matrix_next_layer[new_indexes[i], :]) * (1./(nb_occurences[new_indexes[i]] + 1))
            # Add noise
            if add_noise:
                noise_out = np.random.normal(0, 0.01, new_output_channel.shape)
                new_output_channel += noise_out
                noise_in = np.random.normal(0, 0.01, new_input_channel.shape)
                new_input_channel += noise_in                
        # Add matrix dimensions to be able to concatenate new output channel to weigths matrix
        new_output_channel = new_output_channel[:, np.newaxis]
        new_weights_matrix = np.concatenate((new_weights_matrix, new_output_channel), axis=1)    
        # Add matrix dimensions to be able to concatenate new output channel to weigths matrix
        new_input_channel = new_input_channel[np.newaxis, :]
        new_weights_matrix_next_layer = np.concatenate((new_weights_matrix_next_layer, new_input_channel), axis=0)
    # Divide original layers by duplication factor
    for i in range(len(nb_occurences)):
        new_weights_matrix_next_layer[new_indexes[i], :] *= (1./(nb_occurences[i] + 1))
    return new_weights_matrix, new_biases_vector, new_weights_matrix_next_layer

def net2DeeperNet_CNN(W_U):
    """ Implementation of net2DeeperNet method for a CN layer
    Compute the weights and biases of the new U+1 (UNext) layer based on the weights of layer U

    Args:
      W_U: Weights matrix of layer U. 4D Numpy array of size 
          [filter_height_layer_U, filter_width_layer_U, old_nb_of_input_channels_layer_U, old_nb_of_output_channels_layer_U]

    Returns:
      new_weights_matrix_next_layer: New weights matrix of layer UNext. 4D Numpy array of size 
          [filter_height_layer_U, filter_width_layer_U, old_nb_of_output_channels_layer_U, old_nb_of_output_channels_layer_U]
      new_biases_vector: New biases matrix of layer UNext. 1D Numpy array of size [old_nb_of_output_channels_layer_U]
     """
    # Extract the current number of output channels
    weights_matrix_shape = W_U.shape
    # Initialise weight matrix to build identity matrix
    new_weights_matrix_next_layer = np.zeros((weights_matrix_shape[0], weights_matrix_shape[1], weights_matrix_shape[3], weights_matrix_shape[3]), dtype=np.float32)
    center_h = int((weights_matrix_shape[0]-1)/2)
    center_w = int((weights_matrix_shape[1]-1)/2)
    for idx_channel_in in range(new_weights_matrix_next_layer.shape[2]):
        tmp = np.zeros((weights_matrix_shape[0], weights_matrix_shape[1]), dtype=np.float32)
        tmp[center_h, center_w] = 1
        new_weights_matrix_next_layer[:, :, idx_channel_in, idx_channel_in] = tmp
    # Create biases vector
    new_biases_vector_next_layer = np.zeros(weights_matrix_shape[3], dtype=np.float32)
    return new_weights_matrix_next_layer, new_biases_vector_next_layer

def net2DeeperNet_FC(W_U):
    """ Implementation of net2DeeperNet method for a FC layer
    Compute the weights and biases of the new U+1 (UNext) layer based on the weights of layer U

    Args:
      W_U: Weights matrix of layer U. 2D Numpy array of size 
          [old_nb_of_input_nodes_layer_U, old_nb_of_output_nodes_layer_U]

    Returns:
      new_weights_matrix_next_layer: New weights matrix of layer UNext. 2D Numpy array of size 
          [old_nb_of_output_nodes_layer_U, old_nb_of_output_nodes_layer_U]
      new_biases_vector: New biases matrix of layer UNext. 1D Numpy array of size [old_nb_of_output_nodes_layer_U]
     """    
    # Extract the current number of output channels
    weights_matrix_shape = W_U.shape
    # Initialise weight matrix to build identity matrix    
    new_weights_matrix_next_layer = np.eye(weights_matrix_shape[0])
    # Create biases vector
    new_biases_vector = np.zeros(weights_matrix_shape[0])
    return new_weights_matrix_next_layer, new_biases_vector