import numpy as np
import sys
import os

# Read args
source = sys.argv[1];

# Iterate on every file
for filename in os.listdir(source):
    if "tr_X" in filename:
        X_train = np.load(source + filename)
        print "X_train file : " + filename
    elif "va_X" in filename:
        X_val = np.load(source + filename)
        print "X_val file : " + filename
    elif "_X" in filename: 
        X_test = np.load(source + filename)
        print "X_test file : " + filename
    elif "tr_y" in filename:
        y_tr = np.load(source + filename)
        print "X_val file : " + filename
    elif "va_y" in filename: 
        y_va = np.load(source + filename)
        print "X_test file : " + filename

X_train = X_train[1:5000,:]
X_val = X_val[1:5000,:]
y_tr = y_tr[1:5000,:]
y_va = y_va[1:5000,:]

# Normalization 
max_train = X_train.max()
min_train = X_train.min()
max_val = X_val.max()
min_val = X_val.min()
max_test = X_test.max()
min_test = X_test.min()

"""max_Global = max(max_train, max_val, max_test)
min_Global = min(min_train, min_val, min_test)

X_val_norm = (X_val - min_Global)/(max_Global - min_Global)
X_test_norm = (X_test - min_Global)/(max_Global - min_Global)
X_train_norm = (X_train - min_Global)/(max_Global - min_Global)"""

X_val_norm = (X_val - min_train)/(max_train - min_train)
X_test_norm = (X_test - min_train)/(max_train - min_train)
X_train_norm = (X_train - min_train)/(max_train - min_train)

# Compute the mean
train_mean = np.mean(X_train_norm, axis = 0)

# Substract it
X_train_norm = X_train_norm - train_mean
X_val_norm = X_val_norm - train_mean
X_test_norm = X_test_norm - train_mean

# Get the name
np.save('{}X_train_norm'.format(source + 'normalized/' ), X_train_norm)
np.save('{}X_val_norm'.format(source + 'normalized/' ), X_val_norm)
np.save('{}y_train_norm'.format(source + 'normalized/' ), y_tr)
np.save('{}y_val_norm'.format(source + 'normalized/' ), y_va)




