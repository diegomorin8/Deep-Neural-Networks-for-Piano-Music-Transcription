import numpy as np
import sys
import os
from scipy.io import wavfile
from python_speech_features import mfcc


# Parameters
win_step = 0.01
number_notes = 88;
# Output .npz
output2npz = []
align_vector = []

# Transform to raw data from wav. Get the sampling rate 2
sampling_freq, stereo_vector = wavfile.read("test1.wav")
# Transform to mono
mono_vector = np.mean(stereo_vector, axis = 1)
# Extract mfcc_features
mfcc_feat = mfcc(mono_vector, sampling_freq, winlen = 0.02)
# Number of frames in the file
number_Frames = np.max( mfcc_feat.shape )
# Aux_Vector of times
vector_aux = np.arange(1, number_Frames + 1)*0.01
# Binary labels - we need multiple labels at the same time to represent the chords
labels = np.zeros((number_Frames, number_notes))

# Open the align txt labels
file = open("test1.txt","r")

# Loop over all the lines
for line in file: 
    line_split = line.split()
    if line_split[0] == "OnsetTime":
        print " First line: ignore "
    else:
	# Get the values from the text
	init_range, fin_range, pitch = float(line_split[0]), float(line_split[1]), int(line_split[2])
        # Pitch move to 0-87 range
	pitch = pitch - 21;
        # Get the range indexes
        index_min = np.where(vector_aux >= init_range)
        index_max = np.where(vector_aux - 0.01 > int((fin_range)*100)/float(100))
        labels[index_min[0][0]:index_max[0][0],pitch] = 1


print labels

# Append to add to npz
output2npz.append(mfcc_feat.transpose())



