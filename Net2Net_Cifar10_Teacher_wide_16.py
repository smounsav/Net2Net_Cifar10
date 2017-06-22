# Saypraseuth Mounsaveng
# Net2Net_Cifar10_Teacher_wide_16.py v0.9 01.05.2017
#
# This project aims at testing the Net2Net functions described
# in Net2Net: ACCELERATING LEARNING VIA KNOWLEDGE TRANSFER
# Tianqi Chen , Ian Goodfellow, and Jonathon Shlens arvix 1511.05641
#
# Net2Net_Cifar10_Teacher_wide_16.py contains the teacher CNN that we will use
# in the different experiments of the project with Net2WiderNet
# The teacher CNN is built on following model:
#   - an input layer taking as input images of size 32x32x3 height * width * colors  
#   - a convolutional layer with patch 5x5, stride (1,1), 3 input channels, 16 output channels
#   - a relu activation layer
#   - a max_pool pooling layer with patch 2x2
#   - a normalisation layer
#   - a convolutional layer with patch 5x5, stride (1,1), 16 input channels, 64 output channels
#   - a relu activation layer
#   - a max_pool pooling layer with patch 2x2
#   - a normalisation layer
#   - a fully connected layer with 1024 nodes
#   - an output fully connected layer with 10 nodes, with a dropout of probability 0.5 applied
# ==================================================================
# Import necessary libraries
import numpy as np
import tensorflow as tf

import Net2Net_Cifar10_Input_Toolbox as N2NInTb
import Net2Net_Cifar10_Model_Toolbox as N2NMoTb

# Global variables
BATCH_SIZE = 100 # number of images per mini-batch
NUM_CLASSES = 10 # number of image classes

CHKPT = './teacher-wide-16' # File containing the teacher model
PATH_TENSORBOARD_LOG = './cifar-10-python/cifar-10-logs/teacher-wide-16'

# Test if the Cifar 10 dataset is already present and if not download a local version
N2NInTb.maybe_download_and_extract()

# Extract training data from cifar-10 training files
train_dataset, train_labels = N2NInTb.read_train_data()

# Extract test data from cifar-10 test files
test_dataset, test_labels = N2NInTb.read_test_data()

# Define input and output
with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, 3072], name = 'InputImages') # images
    x_image = tf.reshape(x, [-1,3,32,32]) # index, depth, height, width
    x_image = tf.transpose(x_image,[0,2,3,1]) # Trasnposes index, height, width, depth
#tf.summary.image("Images", x_image)

with tf.name_scope("Groundtruth"):
    y_ = tf.placeholder(tf.int64, [None], name = 'GroundTruth') # labels

# Define network
# Define convolutional layer 1 
with tf.name_scope("ConvLayer1"):
    W_conv1 = N2NMoTb.init_weight([5, 5, 3, 16]) # 5x5 patch, 3 input channels, 16 output channels
    b_conv1 = N2NMoTb.init_bias([16])
    y_conv1 = N2NMoTb.conv2D(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(y_conv1)
tf.summary.histogram("Weights_1", W_conv1)
tf.summary.histogram("Biases_1", b_conv1)
# Define max pooling layer 1
with tf.name_scope("MaxPoolLayer1") as scope:    
    h_pool1 = N2NMoTb.max_pooling_2x2(h_conv1)
# Define normalize layer 1
with tf.name_scope("NormalizeLayer1") as scope:    
    norm1 = tf.nn.lrn(h_pool1, 4)
    
# Define convolutional layer 2
with tf.name_scope("ConvLayer2"):
    W_conv2 = N2NMoTb.init_weight([5, 5, 16, 64]) # 5x5 patch, 16 input channels, 64 output channels
    b_conv2 = N2NMoTb.init_bias([64])
    y_conv2 = N2NMoTb.conv2D(norm1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(y_conv2)
tf.summary.histogram("Weights_2", W_conv2)
tf.summary.histogram("Biases_2", b_conv2)
# Define max pooling layer 2
with tf.name_scope("MaxPoolLayer2") as scope:    
    h_pool2 = N2NMoTb.max_pooling_2x2(h_conv2)
# Define normalize layer 2
with tf.name_scope("NormalizeLayer2") as scope:    
    norm2 = tf.nn.lrn(h_pool2, 4)

# Define fully connected
with tf.name_scope("FC") as scope:
    W_fc1 = N2NMoTb.init_weight([8 * 8 * 64, 1024])
    b_fc1 = N2NMoTb.init_bias([1024])
    h_pool2_flat = tf.reshape(norm2, [-1, 8*8*64])
    y_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(y_fc1)
tf.summary.histogram("Weights_FC1", W_fc1)
tf.summary.histogram("Biases_FC1", b_fc1)

# Apply dropout on last layer
with tf.name_scope("OutputLayer") as scope:
    W_fc2 = N2NMoTb.init_weight([1024, 10])
    b_fc2 = N2NMoTb.init_bias([10])
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
tf.summary.histogram("Weights_Output", W_fc2)
tf.summary.histogram("Biases_Ouput", b_fc2)

# Define loss function
with tf.name_scope("CrossEntropy") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
tf.summary.scalar("Loss", cross_entropy)

# Define optimizer
with tf.name_scope("Train") as scope:    
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train = optimizer.minimize(cross_entropy, global_step=global_step)

# Define accuracy
with tf.name_scope("Accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy", accuracy)

# Define variables to 
VARIABLES_TO_SAVE = {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2, 'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2, 'global_step': global_step }

# Run session
sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(PATH_TENSORBOARD_LOG + '/train', sess.graph)
test_writer = tf.summary.FileWriter(PATH_TENSORBOARD_LOG + '/test')
saver = tf.train.Saver(VARIABLES_TO_SAVE, max_to_keep=None)
tf.global_variables_initializer().run()

#saver.restore(sess, CHKPT)
print("Start learning phase...")
for epoch in range(100001): 
    nb_images = len(train_dataset)
    # Draw random offset to build batch
    offset = np.random.randint(0, nb_images - BATCH_SIZE)
    batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

    train_data_dict = {x: batch_data, y_:  batch_labels, keep_prob: 0.5}
    acc_train_data_dict = {x: batch_data, y_:  batch_labels, keep_prob: 1}
    test_data_dict = {x: test_dataset, y_: test_labels, keep_prob: 1}

    if epoch % 100 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict = acc_train_data_dict)
        train_writer.add_summary(summary, epoch) # Add present statistics to writer
        train_writer.flush() # Force writer to write data to file
        print("step %d, train accuracy %g"%(epoch, acc))
        summary, acc = sess.run([merged, accuracy], feed_dict = test_data_dict)
        test_writer.add_summary(summary, epoch) # Add present statistics to writer
        test_writer.flush() # Force writer to write data to file        
        print("test accuracy %g"%acc)
    if epoch % 1000 == 0:
        save_path = saver.save(sess, CHKPT, global_step=global_step)
    sess.run(train, feed_dict = train_data_dict)
print("End of learning phase")
train_writer.close()
test_writer.close()
sess.close()