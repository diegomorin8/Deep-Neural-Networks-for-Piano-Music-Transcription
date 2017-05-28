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
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Dropout, Activation, Masking
from keras.regularizers import l2
from keras import callbacks
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

mini_batch_size, num_epochs = 100, 1000
input_size = 252
number_units = 256
number_layers = 4
number_classes = 88
best_accuracy = 0
size_samples = 100
contador_bad = 0

#Arg inputs
data_directory = sys.argv[1]
weights_dir = sys.argv[2]

print 'Build model...' 
model = load_model(weights_dir + "weights.hdf5")
history = History()

X = []
y = []

print 'Load validation data...'
X_val = np.load(data_directory + "train_va/" + str(0) + "train_va_X.npy" )
max_shape = (X_val.shape[0]//100)*100
X_val = np.reshape(X_val[0:max_shape,:],(X_val.shape[0]//size_samples,size_samples,X_val.shape[1]))
y_val = np.load(data_directory + "train_va/" + str(0) + "train_va_y.npy" )
max_shape = (y_val.shape[0]//100)*100
y_val = np.reshape(y_val[0:max_shape,:],(y_val.shape[0]//size_samples,size_samples,y_val.shape[1]))
num_tr_batches = len([name for name in os.listdir(data_directory + "train_tr/")])/2

print 'Loading all data'
for i in range(num_tr_batches):
    print "Batching..." + str(i) + "train_tr_X.npy"
    X_train = np.load(data_directory + "train_tr/" + str(i) + "train_tr_X.npy" )
    y_train = np.load(data_directory + "train_tr/" + str(i) + "train_tr_y.npy" )
    max_shape = (X_train.shape[0]//100)*100 
    X_train = np.array(np.reshape(X_train[0:max_shape,:],(X_train.shape[0]//size_samples,size_samples,input_size)))
    y_train = np.array(np.reshape(y_train[0:max_shape,:],(y_train.shape[0]//size_samples,size_samples,number_classes)))
    if i == 0:
        X = X_train
        y = y_train
    else:
        X = np.concatenate((X,X_train), axis = 0)
        y = np.concatenate((y,y_train), axis = 0)

checkpointer = ModelCheckpoint(filepath= weights_dir + "weights.hdf5", verbose=1, save_best_only=False)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

training_log = open(weights_dir + "Training.log", "w")
print 'Train . . .'
# let's say you have an ImageNet generator that yields ~10k samples at a time.
save = model.fit(X, y,epochs = num_epochs,verbose=1,validation_data=(X_val, y_val), callbacks=[checkpointer,early])
training_log.write(str(save.history) + "\n")
training_log.close()
