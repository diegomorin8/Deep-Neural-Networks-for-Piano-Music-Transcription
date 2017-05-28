import numpy as np
import sys
import os
from scipy.io import wavfile
from python_speech_features import mfcc

# Read args
Label_text_source = sys.argv[1];
Output_dir = sys.argv[2];

f = open(Output_dir + 'train.lst','w')

for filename in os.listdir(Label_text_source):
    f.write(filename + '\n')

f.close() 

