## Introduction
+ The project is a simple implement for LeNet-5 using TensorFlow, training with MNIST

## LeNet-5
+ LeNet-5 is from the paper 《Gradient-Based Learning Applied to Document Recognition》. It contains 7 layers, including 5 convolution layers and 2 fully connection layers. The below figure shows the model structure:
![LeNet](./images/lenet.png)
+ The overall structure is: input layer --> convolutional layer --> pooling layer --> activation function --> convolutional layer --> pooling layer --> activation function --> fully connect layer --> activation function --> fully connect layer --> activation function --> output
+ input layer: ? * 28 * 28 * 1 --> ? * 32 * 32 * 1
+ conv1 layer: ? * 32 * 32 * 1 --> ? * 28 * 28 * 6
+ pooling layer: ? * 28 * 28 * 6 --> ? * 14 * 14 * 6
+ conv2 layer: ? * 14 * 14 * 7 --> ? * 10 * 10 * 16
+ pooling layer: ? * 10 * 10 * 16 --> ? * 5 * 5 * 16
+ fc1: ? * 5 * 5 * 16 --> ? * 120
+ fc2: ? * 120 --> ? * 84
+ output: ? * 84 --> ? * 10


