# Net2Net_Cifar10
Implementation of Net2Net method and application on Cifar10 dataset

Introduction

The Net2Net method proposes a way to transfer the knowledge acquired by a neural network (called the teacher) to a bigger one (called the student) to accelerate the learning of the latter.
Net2Net is composed of 2 functions: Net2WiderNet and Net2DeeperNet.
The first one, Net2WiderNet, is used to transfer the knowledge of a layer to a wider one. A wider fully connected layer is a layer with more nodes. A wider convolutional layer is layer with more output channels. In Net2WiderNet, instead of being initialized randomly, the new nodes or convolutional channels are copied from existing nodes/channels. If the extended layer has a following layer in the network model, this following layer is also extended by the number of new nodes or convolutional channels, also by replicating existing nodes or convolutional channels. Moreover, after the extension, the weights of the nodes or convolutional channels of the following layer with the same weights are divided by the number of similar nodes or output channels. This ensures that the student network, with the extended layer, keeps an accuracy  close to the one of the teacher network.
The second function, Net2DeeperNet, delivers a new layer L2 based on a defined layer L1 and inserted just after L1 in the model of the student network. L2 is initialized in a way that given an input i of L1, a function fL1 associated to layer L1 and a function fL2 associated to L2, we have fL1(i) =fL2(fL1(i)). An example of  function with this property is the identity function, also referred to in other papers as IdMorph.

Project description

Different papers mentioned the difficulty to train very big networks from scratch. An approach to respond to this problem could be, instead of training the big network directly with the final size from scratch, to train first a smaller network, called the teacher, and then to extend it gradually into bigger networks, called student networks, until the last student network reaches the final size.
In this project, we will experiment this gradual extension approach by using Net2Net to transfer the knowledge of the teacher network to its student after each extension. The network referred to as reference network will be the network with the final size.

The experimentation will be divided in 3 phases. 
First, we will implement and test the Net2Net functions separately.
Second, we will focus on going deeper by testing if it's faster to train an 8 layer network by making it progressively deeper or by training it directly with the final size and design.
Finally, we will push further the experimentation of the second part by training a much bigger network (20 layers).

The implementation will be done using the Python language and the Tensorflow Deep Learning library.

The networks trained will be convolutional neural networks and they will be used to classify images from the Cifar-10 dataset.
As a reminder, the Cifar-10 dataset is a collection of 60000 32x32 color images distributed in 10 classes, with 6000 images per class. This collection is divided in 2 parts: a training dataset of 50000 images, and a test dataset of 10000 images. 
The training will be done using the stochastic gradient descent method with a loss based on the cross-entropy between the image labels and the logits of the network normalized with softmax.
The size of a mini-batch is 100 images.

Reference

Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2Net: ACCELERATING LEARNING VIA KNOWLEDGE TRANSFER. Arvix: 1511.05641
