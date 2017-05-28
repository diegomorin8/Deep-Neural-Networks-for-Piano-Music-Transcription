import numpy as np
import sys
import os
import matplotlib.pyplot as plt
# Read args
source = sys.argv[1]
train_folder = "train_tr/"
val_folder = "train_va/"
test_folder = "test/"

mean_X = []
min_X = []
max_X = []

print "Get max - min "
# Iterate on every file
for filename in os.listdir(source + train_folder):
    if "tr_X" in filename:
        X_train = np.load(source + train_folder + filename)
        max_X.append(X_train.max())
        min_X.append(X_train.min())
        
max_train = max(max_X)
min_train = min(min_X)

print "Get mean"
total_length = 0
# Iterate on every file
for filename in os.listdir(source + train_folder):
    if "tr_X" in filename:
        X_train = np.load(source + train_folder + filename)
        X_train_norm = (X_train - min_train)/(max_train - min_train)
        # Compute the mean
        mean_X.append(np.sum(X_train_norm, axis = 0))
        total_length = total_length + len(X_train_norm)

train_mean = np.sum(mean_X, axis = 0)/float(total_length)

print "Normalize "
# Iterate on every file
for filename in os.listdir(source + train_folder):
    filename_split = filename.split('.')
    if "tr_X" in filename:
        X_train = np.load(source + train_folder + filename)
        X_train_norm = (X_train - min_train)/(max_train - min_train) 
        X_train_norm = X_train_norm - train_mean
        print "X_train file : " + filename
        np.save('{}'.format(source + train_folder + filename_split[0] ), X_train_norm)

for filename in os.listdir(source + val_folder):
    filename_split = filename.split('.')
    if "va_X" in filename:
        X_val = np.load(source + val_folder+ filename)
        X_val_norm = (X_val - min_train)/(max_train - min_train)
        X_val_norm = X_val_norm - train_mean
        print "X_val file : " + filename
        np.save('{}'.format(source + val_folder + filename_split[0]), X_val_norm)
    
for filename in os.listdir(source + test_folder):
    filename_split = filename.split('.')
    if "_X" in filename: 
        X_test = np.load(source + test_folder + filename)  
        X_test_norm = (X_test - min_train)/(max_train - min_train) 
        X_test_norm = X_test_norm - train_mean
        print "X_test file : " + filename
        np.save('{}'.format(source + test_folder + filename_split[0] ), X_test_norm) 
    

print train_mean
print min_train
print max_train


