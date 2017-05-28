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
size_samples = 100
data_directory = sys.argv[1]
weights_dir = sys.argv[2]

X = []
y = []

num_test_batches = len([name for name in os.listdir(data_directory )])/2

print 'Loading test data'
for i in range(num_test_batches):
    print "Batching..." + str(i) + "test_X.npy"
    X_test = np.load(data_directory + str(i) + "test_X.npy" )
    y_test = np.load(data_directory + str(i) + "test_y.npy" )
    max_shape = (X_test.shape[0]//100)*100 
    X_test = np.array(np.reshape(X_test[0:max_shape,:],(X_test.shape[0]//size_samples,size_samples,input_size)))
    y_test = np.array(np.reshape(y_test[0:max_shape,:],(y_test.shape[0]//size_samples,size_samples,number_classes)))
    if i == 0:
        X = X_test
        y = y_test
    else:
        X = np.concatenate((X,X_test), axis = 0)
        y = np.concatenate((y,y_test), axis = 0)

# Load the model 
model = load_model(weights_dir + "weights.hdf5")
TP = 0
FP = 0
FN = 0

print "Predicting model. . . "
predictions = model.predict(X, batch_size=mini_batch_size, verbose = 1) 
predictions = np.reshape(predictions,(y.shape[0]*y.shape[1],y.shape[2]))
y = np.reshape(y,(y.shape[0]*y.shape[1],y.shape[2]))
predictions = np.array(predictions).round()
predictions[predictions > 1] = 1
np.save('{}predictions'.format(weights_dir), predictions)

print "\nCalculating accuracy. . ."
TP = np.count_nonzero(np.logical_and( predictions == 1, y == 1 ))
FN = np.count_nonzero(np.logical_and( predictions == 0, y == 1 ))
FP = np.count_nonzero(np.logical_and( predictions == 1, y == 0 ))
if (TP + FN) > 0:
    R = TP/float(TP + FN)
    P = TP/float(TP + FP)
    A = 100*TP/float(TP + FP + FN)
    if P == 0 and R == 0:
	F = 0
    else: 
	F = 100*2*P*R/(P + R)
else: 
    A = 0
    F = 0
    R = 0
    P = 0

print '\n F-measure pre-processed: '
print F
print '\n Accuracy pre-processed: '
print A

print "\nCleaning model . . ."
for a in range(predictions.shape[1]):
    for j in range(2,predictions.shape[0]-3):
        if predictions[j-1,a] == 1 and predictions[j,a] == 0 and predictions[j+1,a] == 0 and predictions[j+2,a] == 1:
            predictions[j,a] = 1
            predictions[j+1,a] = 1
        if predictions[j-2,a] == 0 and predictions[j-1,a] == 0 and predictions[j,a] == 1 and predictions[j+1,a] == 1 and predictions[j+2,a] == 0 and predictions[j+3,a] == 0:
            predictions[j,a] = 0
            predictions[j+1,a] = 0
        if predictions[j-1,a] == 0 and predictions[j,a] == 1 and predictions[j+1,a] == 0 and predictions[j+2,a] == 0:
            predictions[j,a] = 0
        if predictions[j-1,a] == 1 and predictions[j,a] == 0 and predictions[j+1,a] == 1 and predictions[j+2,a] == 1:
            predictions[j,a] = 1

print "Calculating accuracy after cleaning. . ."
np.save('{}predictions_post'.format(weights_dir), predictions)
TP = np.count_nonzero(np.logical_and( predictions == 1, y == 1 ))
FN = np.count_nonzero(np.logical_and( predictions == 0, y == 1 ))
FP = np.count_nonzero(np.logical_and( predictions == 1, y == 0 ))
if (TP + FN) > 0:
    R = TP/float(TP + FN)
    P = TP/float(TP + FP)
    A = 100*TP/float(TP + FP + FN)
    if P == 0 and R == 0:
	F = 0
    else: 
	F = 100*2*P*R/(P + R)
else: 
    A = 0
    F = 0
    R = 0
    P = 0

print '\n F-measure post-processed: '
print F
print '\n Accuracy post-processed: '
print A


main_data = open(weights_dir + "Accuracy.lst", "w")
main_data.write("R-pre = " + str("%.6f" % R) + "\n")
main_data.write("P-pre = " + str("%.6f" % P) + "\n")
main_data.write("A-pre = " + str("%.6f" % A) + "\n")
main_data.write("F-pre = " + str("%.6f" % F) + "\n")
main_data.write("R-post = " + str("%.6f" % R) + "\n")
main_data.write("P-post = " + str("%.6f" % P) + "\n")
main_data.write("A-post = " + str("%.6f" % A) + "\n")
main_data.write("F-post = " + str("%.6f" % F) + "\n")
main_data.close()

