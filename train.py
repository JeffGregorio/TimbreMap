import os
import sys
import argparse
from util.dataset import *
from util.models import *
from util.tests import *

# Create parser for command line arguments
parser = argparse.ArgumentParser(description='Train TimbreMap models')

# Positional arguments 
parser.add_argument('data_dir', help='data directory')
parser.add_argument('model_dir', help='model directory')

# Mutually-exclusive group (one required)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dnn', action='store_true', help='use dnn encoder')
group.add_argument('--cnn', action='store_true', help='use cnn encoder')
group.add_argument('--lstm', action='store_true', help='use lstm encoder')

# Optional arguments
parser.add_argument('--gen', action='store_true', help='use generative latent encoding')
parser.add_argument('--pca', action='store_true', help='use principal component analysis')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--latent_size', type=int, default=3,  help='latent size')

# Parse
args = parser.parse_args()

# Verify data directory exists
if not os.path.exists(args.data_dir):
	print("Data directory \"%s\" does not exist" % args.data_dir)
	sys.exit()

# Make model directory if it doesn't exist
if not os.path.exists(args.model_dir):
	os.makedirs(args.model_dir)

# =====================
# Load/Preprocess Data:
# =====================

# Load training and testing partitions
(x_train, y_train), (x_test, y_test) = load_data(args.data_dir)

# Standardize
x_train, x_test = standardize(x_train, x_test)

# =============
# Build Models:
# =============

# Build specified encoder model
if args.dnn:
	numel = np.prod(x_train.shape[1:])
	encoder = build_encoder_dnn(
		input_shape=x_train.shape[1:], 
		latent_size=args.latent_size,
		dense_sizes=(numel//4, numel//16),
		generative=args.gen)

elif args.cnn:
	# Add an explicit channel dimension to our grayscale images so the data has 
	# shape (batch, height, width, channels)
	x_train = np.reshape(x_train, x_train.shape + (1,))
	x_test = np.reshape(x_test, x_test.shape + (1,))
	encoder = build_encoder_cnn(
		input_shape=x_train.shape[1:], 
		latent_size=args.latent_size,
		generative=args.gen)

elif args.lstm:
	# Transpose examples (Keras LSTMs have shape (example, timestep, feature))
	x_train = np.swapaxes(x_train, 1, 2)
	x_test = np.swapaxes(x_test, 1, 2)
	encoder = build_encoder_lstm(
		input_shape=x_train.shape[1:], 
		latent_size=args.latent_size,
		lstm_sizes=(128,),
		generative=args.gen)

# Build regressor and assemble end-to-end model
regressor = build_regressor(
	latent_size=args.latent_size, 
	output_size=y_train.shape[1])
model = build_end_to_end(encoder, regressor)


# ===========
# Train/Eval:
# ===========

# Train
model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch, shuffle=True)

# Evaluate and export error plots
model_eval(model, x_test, y_test, args.model_dir)

# ===================
# Export Keras Model:
# ===================

# Save models (Keras's JSON and H5 formats)
export_keras(
	model_dir=os.path.join(args.model_dir, 'keras'), 
	model=model, 
	encoder=encoder, 
	regressor=regressor)

# =======================
# Export TimbreMap Model:
# =======================

# Encode and save latent space data
latent = encoder.predict(np.append(x_train, x_test, axis=0))
np.save(os.path.join(args.model_dir, 'latent'), latent)

# Export the regressor model parameters
p_dir = os.path.join(args.model_dir, 'timbremap')
export_regressor(p_dir, regressor)

# Export latent space means and variances
if not args.pca:
	export_vec_scale(p_dir, latent)
	print("Testing (c -> z -> p) -> (p -> z -> c)");

# Or perform PCA and export basis vectors and biases, plus re-oriented 
# latent space statistics
else:
	weights, biases, latent_pca = pca(latent)
	np.save(os.path.join(args.model_dir, 'latent_pca'), latent_pca)
	export_vec_scale(p_dir, latent_pca)
	export_pca_layer(p_dir, weights, biases)
	print("Testing (c -> z' -> z -> p) -> (p -> z -> z' -> c)");

# Verify forward and inverse mapping invertibility
print("Error: %f" % test_max(p_dir))

	