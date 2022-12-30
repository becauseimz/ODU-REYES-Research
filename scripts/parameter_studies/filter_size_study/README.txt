This directory summarizes the studies done
to determine the depencence of the neural network
predictive power on the filter size used in the convolution.

A plot is made of the ratio: correct_predictions / (total images )
versus each of the focal plane plots, for each of the filter_size
configurations

This study serves to study the dependence of the model predictive
power  due to changes in the number of filters used



In the keras_optics.py script, the following parameters were used:

fixed parameters:
------------------
num_filters = 12
pool_size = 2
input_image=(200,200,1)
activation='softmax'
epochs=100

optimizer='adam'
loss='categorical_crossentropy'
metrics=['accuracy']


variable parameters:
---------------------
num_filters = 1, 6 or 12 ( filter_size = 3 config exists from previous study of num_filters)



Directory Structure:
--------------------
./Study1 : optics weights files and plots using filter_size = 1
./Study2 : optics weights files and plots using filter_size = 6
./Study3 : optics weights files and plots using filter_size = 12


Results: The filter size did not affect the accuracy much, except for the highest filter size, 12x12,
in which the accuracy of one of the 2D plots dripped dramatically, and overall, it was more difficult to
achieve 100 % accuracy for most of the models.  The best or optimum filter size was determined to be
filter_size = 6.  This is NOT much differently from filter_size=3, however, there is one 2D focal plane plot
in which the accuracy improved when going from 3 to 6. 
