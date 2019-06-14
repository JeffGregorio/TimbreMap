import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util.dataset import *
from util.models import *
from util.tests import *

# Create parser for command line arguments
parser = argparse.ArgumentParser(
	description='Plot TimbreMap model latent space distributions')

# Positional arguments 
parser.add_argument('model_dir', help='model directory')

# Parse
args = parser.parse_args()

# Verify data directory exists
if not os.path.exists(args.model_dir):
	print("Model directory \"%s\" does not exist" % args.model_dir)
	sys.exit()

# Load the latent space data
latent = np.load(os.path.join(args.model_dir, 'latent.npy'))
try:
	latent_pca = np.load(os.path.join(args.model_dir, 'latent_pca.npy'))
	scatter_latent(latent, z_projected=latent_pca)
except:
	scatter_latent(latent)



