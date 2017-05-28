'''###### TRAIN 1: DNN - 3 layers - 150 unis per layer ######'''

import numpy as np
import os
import os.path
import sys
import matplotlib.pyplot as plt

# We need to set the random seed so that we get ther same results with the same parameters
np.random.seed(400)  

# Import keras main libraries
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2

mini_batch_size, num_epochs = 100, 50
input_size = 252
number_units = 256
number_layers = 3
number_classes = 88
data_directory = sys.argv[1]
weights_dir = sys.argv[2]

predictions_draw = []
y_draw = []
print 'Predict . . . '
num_test_batches = len([name for name in os.listdir(data_directory)])/2

y = []
print 'Loading test data'
for i in range(num_test_batches):
    print "Batching..." + str(i) + "test_X.npy"
    y_test = np.array(np.load(data_directory + str(i) + "test_y.npy" ))
    if i == 0:
        y = y_test
    else:
        y = np.concatenate((y,y_test), axis = 0)


predictions = np.load(weights_dir + "predictions_post.npy" ) 

plt.figure()
plt.subplot(211)
plt.imshow(predictions.transpose(),cmap='Greys',aspect='auto', interpolation = 'nearest')
plt.figure()
plt.subplot(212)
plt.imshow(y.transpose(),cmap='Greys',aspect='auto',interpolation = 'nearest')
plt.show()
