# Saypraseuth Mounsaveng
# Net2Net_Cifar10_Input_Toolbox.py v0 16.04.2017
#
# This project aims at testing the Net2Net functions described
# in Net2Net: ACCELERATING LEARNING VIA KNOWLEDGE TRANSFER
# Tianqi Chen , Ian Goodfellow, and Jonathon Shlens arvix 1511.05641
#
# Net2Net_Cifar10_Input_Toolbox.py contains functions to read data from the
# Cifar10 dataset files and return them in numpy arrays format:
#    maybe_download_and_extract
#    extract_label_names
#    unpickle
#    read_train_data
#    read_test_data
#
# ==================================================================

import glob
import os
import pickle
import sys
import tarfile

import numpy as np

from six.moves import urllib

# Define global variables
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DEST_DIR = './cifar-10-python/'
EXTRACTED_PATH = 'cifar-10-batches-py/'

LABELS_FILENAME = 'batches.meta'
TEST_FILENAME = 'test_batch'
TRAIN_FILENAME =  'data_batch*'

def maybe_download_and_extract():
    """Download and extract the tarball from Alex Krizhevsky's website.
  
    Args:
      None  
  
    Returns:
      None      
    """
    dest_directory = DEST_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, EXTRACTED_PATH)
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def extract_label_names():
    """ Read batches.meta.txt from python dataset and return label names
  
    Args:
      None  
  
    Returns:
      labels_name: List of label names. List
    """
    dest_directory = DEST_DIR
    if not os.path.exists(dest_directory):
        print('Error: cifar 10 path not existing')
        return
    else:
        extracted_dir_path = os.path.join(dest_directory, EXTRACTED_PATH)
        if not os.path.exists(extracted_dir_path):
            print('Error: cifar 10 extracted path not existing')
            return
        else:
            labels_filepath = os.path.join(extracted_dir_path, LABELS_FILENAME)
            if not os.path.exists(labels_filepath):
                print('Error: cifar 10 labels file not existing')
                return              
            else:
                labels_name = []
                with open(labels_filepath , 'r') as f:
                    labels = f.read().splitlines()
                    labels = filter(None, labels) # Remove empty lines
                    for label in labels:
                        labels_name.append(label)
                    return labels_name
     
def unpickle(file):
    """ Read data from input file
    
    Args:
      file: Input file containing dataset. File  
  
    Returns:
      dict: Dictionary containing data and corresponding labels. Dictionary.
    """    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_train_data():
    """ Read data from training dataset file and return training examples list
    
    Args:
      None 
  
    Returns:
      train_images : List of images. List
      train_labels: List of corresponding labels. List
    """
    train_data_path = os.path.join(DEST_DIR, EXTRACTED_PATH)
    list_data_files = glob.glob(train_data_path + TRAIN_FILENAME)
    train_images = np.empty((0))
    train_labels = np.empty((0))
    for file in list_data_files:
        train_data = unpickle(file)
        if len(train_images) == 0:
            train_images = train_data[b'data']
        else:
            train_images = np.concatenate((train_images, train_data[b'data']), axis=0)
        if len(train_labels) == 0:
            train_labels = train_data[b'labels']
        else:
            train_labels = np.concatenate((train_labels, train_data[b'labels']), axis=0)
    train_images = train_images.astype(float)
    train_labels = np.array(train_labels)
    # Shuffle dataset
    perm = np.random.permutation(len(train_images))
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    return train_images, train_labels

def read_test_data():
    """ Read data from training dataset file and return training examples list
    
    Args:
      None 
  
    Returns:
      train_images : List of images. List
      train_labels: List of corresponding labels. List
    """    
    test_data_path = os.path.join(DEST_DIR, EXTRACTED_PATH)
    test_images = np.empty((0))
    test_labels = np.empty((0))
    test_data = unpickle(test_data_path + TEST_FILENAME)
    test_images = test_data[b'data']
    test_labels = test_data[b'labels']
    
    return test_images, test_labels