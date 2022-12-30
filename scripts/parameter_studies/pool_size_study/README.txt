This directory summarizes the studies done
to determine the depencence of the neural network
predictive power on the pool size used in the convolution.

A plot is made of the ratio: correct_predictions / (total images )
versus each of the focal plane plots, for each of the pool_size
configurations

This study serves to study the dependence of the model predictive
power  due to changes in the pool size used


In the keras_optics.py script, the following parameters were used:

fixed parameters:
------------------
num_filters = 12
filter_size = 6
input_image=(200,200,1)
activation='softmax'
epochs=100

optimizer='adam'
loss='categorical_crossentropy'
metrics=['accuracy']


variable parameters:
---------------------
pool_size = 1, 4, 6, 8  



Directory Structure:
--------------------
./Study1 : optics weights files and plots using pool_size = 1
./Study2 : optics weights files and plots using pool_size = 4
./Study3 : optics weights files and plots using pool_size = 6
./Study4 : optics weights files and plots using pool_size = 8

Results: The optimum pool_size parameter was determined to be pool_size = 6,
