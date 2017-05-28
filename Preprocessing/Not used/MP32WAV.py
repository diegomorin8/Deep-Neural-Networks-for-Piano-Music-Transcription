# Imports
import sys
import os
from pydub import AudioSegment

# Read args
source_MP3 = sys.argv[1];
out_WAV = sys.argv[2];

for filename in os.listdir(source_MP3):
    sound = AudioSegment.from_file(source_MP3 + filename, "mid")
    # We need to extract mp3
    filename_out = filename.split('.')
    sound.export(out_WAV + filename_out[0] + ".wav", format="wav")



