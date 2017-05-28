'''###### TRAIN 1: DNN - 3 layers - 150 unis per layer ######'''

import numpy as np
import os
import os.path
import sys

# We need to set the random seed so that we get ther same results with the same parameters
np.random.seed(400)  

# Import keras main libraries
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras import callbacks
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

mini_batch_size, num_epochs = 100, 1000
input_size = 252
number_units = 256
number_layers = 3
number_classes = 88
best_accuracy = 0
contador_bad = 0

#Arg inputs
data_directory = sys.argv[1]
weights_dir = sys.argv[2]

print 'Build model...' 
model = Sequential()
history = History()

print 'Load validation data...'
X_val = np.load(data_directory + "train_va/" + str(0) + "train_va_X.npy" )
y_val = np.load(data_directory + "train_va/" + str(0) + "train_va_y.npy" )

# Count the number of files in the training folder 
num_tr_batches = len([name for name in os.listdir(data_directory + "train_tr/")])/2

print 'Loading all data'
for i in range(num_tr_batches):
    print "Batching..." + str(i) + "train_tr_X.npy"
    X_train = np.array(np.load(data_directory + "train_tr/" + str(i) + "train_tr_X.npy" ))
    y_train = np.array(np.load(data_directory + "train_tr/" + str(i) + "train_tr_y.npy" ))
    if i == 0:
        X = X_train
        y = y_train
    else:
        X = np.concatenate((X,X_train), axis = 0)
        y = np.concatenate((y,y_train), axis = 0)

print X.shape
print "Adding 1st layer of {} units".format(number_units)
model.add(Dense(number_units, input_shape=(input_size,), kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
for i in range(number_layers-1):
    print "Adding %d" % (i+2) + "th layer of %d" % number_units + " units"
    model.add(Dense(number_units, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))

print " Adding classification layer"
model.add(Dense(number_classes, kernel_initializer='normal', activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

checkpointer = ModelCheckpoint(filepath= weights_dir + "weights.hdf5", verbose=1, save_best_only=False)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

training_log = open(weights_dir + "Training.log", "w")
print 'Train . . .'
# let's say you have an ImageNet generat        print "Fitting the batch :"
save = model.fit(X, y,batch_size=mini_batch_size,epochs = num_epochs,validation_data=(X_val, y_val),verbose=1,callbacks=[checkpointer,early])
training_log.write(str(save.history) + "\n")
training_log.close()
