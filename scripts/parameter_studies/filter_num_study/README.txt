This directory summarizes the studies done
to determine the depencence of the neural network
predictive power on the number of filters used in the convolution.

A plot is made of the ratio: correct_predictions / (total images )
versus each of the focal plane plots, for each of the filter_number
configurations

This study serves to study the dependence of the model predictive
power  due to changes in the number of filters used



In the keras_optics.py script, the following parameters were used:

fixed parameters:
------------------
filter_size = 3
pool_size = 2
input_image=(200,200,1)
activation='softmax'
epochs=100

optimizer='adam'
loss='categorical_crossentropy'
metrics=['accuracy']


variable parameters:
---------------------
num_filters = 4, 12 or 24 ( num_filters = 8 exists from the previous study on epochs)



Directory Structure:
--------------------
./Study1 : optics weights files and plots using num_filters = 4
./Study2 : optics weights files and plots using num_filters = 12
./Study3 : optics weights files and plots using num_filters = 24


Results: The studeis showed a slight improvement when num_filters = 12 was used as compared to 8 or 4
When num_filters = 24 was used, it generated the EXACT same results as 12, so we will use num_filters = 12 as optimum
parameter (and epochs=100, as determined from epochs_study)
