import os
import sys
import argparse
import librosa
import numpy as np
from natsort import natsorted
from util.dataset import compute_features

# Feature function supplied to compute_features (in util.dataset)
def compute_melspec(samples, fs):
	return librosa.feature.melspectrogram(
		y=samples, 
		sr=fs,
		n_fft=2048,
		hop_length=128,
		power=2)

# Main
# -------------------------------------------------------------------------- #
# Parser for data directory argument
parser = argparse.ArgumentParser(description='Compute Mel-scaled spectrogram features')
parser.add_argument('data_dir', help='data directory')
args = parser.parse_args()

# Verify data directory exists
if not os.path.exists(args.data_dir):
	print("Data directory \"%s\" not found", args.data_dir)
	sys.exit()

# Verify wavs directory exists
wav_dir = os.path.join(args.data_dir, 'wavs')
if not os.path.exists(wav_dir):
	print("Wavs sub-directory \"%s\" not found", wav_dir)
	sys.exit()

# Get list of files in the wav directory
wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if not f.startswith('.')]
wav_files = natsorted(wav_files)	# Sort by file name (names should be ex_#)

# Compute a numpy array of mel spectrograms of standardized width
melspecs = compute_features(wav_files, compute_melspec, equal_width=True)

# Save
print("Saving features...\n")
np.save(os.path.join(args.data_dir, 'features'), melspecs)

