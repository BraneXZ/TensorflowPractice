# Class recap:
# - We learned about how neural network works with its sequential model
# - A Dense layer performs a dot product between the input and a set of weights
# - The size of weights in a Dense layer is determined by the number of neurons within that layer +
#   bias for each neuron, and the number of features in the previous layer
#   i.e. our exampled had 4 features (BlueGoldPerMin, BlueDeath, BlueKill, RedGoldPerMin)
#   if our Dense layer contains 4 neurons, then we'd have (4 features * (4 + 1)) = 20 parameters to learn
# - An activation function performs non-linearity to the output of a neuron to learn a more complicated function
# - We learned about ReLU (which we'll use for the majority of our hidden layers), sigmoid (for binary outputs)
# - By default, a Dense layer uses linear activation which could be useful for regression tasks
# - After our input gets passed through each layer within our model,
#   it will measure its performance using a loss function
# - The loss function tells the network which direction to go when performing gradient descent during back-propagation
# - Metrics is used primarily for humans to understand the model's performance since
#   loss function outputs a scalar value that isn't interpretable
# - Back-propagation is done using chain rule from calculus to calculate gradients of each layers starting from the
#   output and work backward to update all the weights in the model
# - Optimizer controls how weights are updated during back-propagation. This could slow or speed up convergence
#   depending on the problem and how you choose the parameters (learning_rate, momentum, decay, etc)
#
# - Convolution layers is similar to Dense layer but instead of vectors, you have matrices that slide across the input
# - This sliding window will result in a reduction in dimensionality, which it why often time padding will be used
# - Number of filters controls the output channel within that layer while kernel size controls how big the sliding
#   window is
# - You can also change strides  to slide across the input much quicker which results in even smaller output dimension

from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
import numpy as np
from keras.metrics import Accuracy, Precision
from keras.optimizers import Adam

#
# model = Sequential()
# model.add(Dense(20, activation="relu", input_shape=(4,)))
# model.add(Dense(5))
# model.add(Dense(1, activation="sigmoid"))
#
# model.compile(optimizer="SGD",
#               loss="binary_crossentropy",
#               metrics=["acc"])
#
# model.summary()


model = Sequential()
model.add(Conv2D(72, 3, input_shape=(7, 7, 3)))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy")

model.summary()