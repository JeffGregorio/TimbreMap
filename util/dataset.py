import os
import soundfile
import numpy as np
from natsort import natsorted

# Feature Computation
# ===================
#
# Apply feature_func to every wav file specified by an array of wav file paths.
# Return a numpy array. If feature_func returns images of varying width, the 
# return array will be standardized so that the width of every image is the 
# median width across the dataset, unless specified. 
def compute_features(wav_files, feature_func, equal_width=True):
	features = []
	n = len(wav_files)
	# Load each wav example and compute its mel spectrogram
	for idx, f in enumerate(wav_files):
		print('%6d/%6d: %s' % (idx+1, n, f))
		samples, fs = soundfile.read(f)
		features.append(feature_func(samples, fs))
	if equal_width:
		# Return as numpy array with all images standardized to the median width
		width = int(np.median([img.shape[1] for img in features]))
		return image_list_to_np_array(features, width)
	else:
		# Or a python list of images with (possibly) varying widths	
	 	return features

# Convert python list of images to numpy array with specified width by truncating
# or zero-padding images
def image_list_to_np_array(images, width):
	np_images = np.zeros((len(images), images[0].shape[0], width))
	for idx, img in enumerate(images):
		img_w = min(width, img.shape[1])
		np_images[idx, :, :img_w] = img[:, :img_w]
	return np_images

# Data Preprocessing
# ==================
#
# Add AWGN to an N-D array
def add_noise(x, mu, var):
	n, m, k = x.shape
	x_noise = np.random.normal(mu, var**0.5, (n, m, k))
	return x + x_noise

# Subtract column means and divide by column standard deviations
def standardize(x_train, x_test):
	mu = x_train.mean(axis=(0, 1))
	sd = x_train.std(axis=(0, 1)) + np.finfo(np.float32).eps
	x_train -= mu
	x_train /= sd
	x_test -= mu
	x_test /= sd
	return x_train, x_test

# Standardize across frequency bins
def standardize_freqs(x_train, x_test=None):
	mu = x_train.mean(axis=(0, 2))
	sd = x_train.std(axis=(0, 2)) + np.finfo(np.float32).eps
	for i, ex in enumerate(x_train):
		x_train[i] = (ex.transpose() - mu).transpose()
		x_train[i] = (ex.transpose() / sd).transpose()
	if x_test is not None:
		for i, ex in enumerate(x_test):
			x_test[i] = (ex.transpose() - mu).transpose()
			x_test[i] = (ex.transpose() / sd).transpose()
	return x_train, x_test

# Standardize across frequency bins for tuples of training and testing groups
def standardize_groups(train_groups, test_groups):
	# Reshape the training groups (concatenate along time axis) to easily take the 
	# statistics of each frequency bin
	group_Xtrain = tuple()
	for i in range(len(train_groups)):
		group_Xtrain += (train_groups[i][0].reshape((128, -1)), )
	x_train = np.hstack(group_Xtrain)
	mu = x_train.mean(axis=1)
	sd = x_train.std(axis=1) + np.finfo(np.float32).eps
	for i in range(len(train_groups)):
		for j in range(len(train_groups[0][0])):
			train_groups[i][0][j] = (train_groups[i][0][j].transpose() - mu).transpose()
			train_groups[i][0][j] = (train_groups[i][0][j].transpose() / sd).transpose()
		for j in range(len(test_groups[0][0])):
			test_groups[i][0][j] = (test_groups[i][0][j].transpose() - mu).transpose() 
			test_groups[i][0][j] = (test_groups[i][0][j].transpose() / sd).transpose() 
	return train_groups, test_groups

# # Data Generator
# # ==============
# # Adapted from tutorial at
# # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# class DataGenerator:

# 	def __init__(self, list_IDs, labels, batch_size=32, dim=None, 
# 		n_channels=1, n_classes=10, shuffle=True, data_dir='data'):
# 		'Initialization'
# 		self.dim = dim
# 		self.batch_size = batch_size
# 		self.labels = labels
# 		self.list_IDs = list_IDs
# 		self.n_channels = n_channels
# 		self.n_classes = n_classes
# 		self.shuffle = shuffle
# 		self.on_epoch_end()

# 	def on_epoch_end(self):
# 		'Updates indexes after each epoch'
# 		self.indexes = np.arange(len(self.list_IDs))
# 		if self.shuffle == True:
# 			np.random.shuffle(self.indexes)

# 	def __data_generation(self, list_IDs_temp):
# 		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
# 		# Initialization
# 		X = np.empty((self.batch_size, *self.dim, self.n_channels))
# 		y = np.empty((self.batch_size), dtype=int)

# 		# Generate data
# 		for i, ID in enumerate(list_IDs_temp):
# 			# Store sample
# 			X[i,] = np.load(os.path.join(data_dir, ID + '.npy'))
# 			# Store class
# 			y[i] = self.labels[ID]

# 		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def load_data_varlen(data_dir, test_ratio=0.1):
	group = 0
	train_groups = list()
	test_groups = list()
	while True:
		try:
			x = np.load(os.path.join(data_dir, 'features_%d.npy' % group))
			y = np.load(os.path.join(data_dir, 'labels_%d.npy' % group))
			n = len(x)
			n_test = round(n * test_ratio)

			# Load an existing partition 
			try:
				p = np.load(os.path.join(data_dir, 'partition_%d.npy' % group))
			
			# Or generate and save a new one
			except FileNotFoundError:
				p = np.random.permutation(n)
				np.save(os.path.join(data_dir, 'partition_%d.npy' % group), p)
			
			# Partitioned training and testing sets
			x_train = x[p[n_test:]]
			y_train = y[p[n_test:]]
			x_test = x[p[:n_test]]
			y_test = y[p[:n_test]]
			print("loaded group %d" % group)

			train_groups.append((x_train, y_train))
			test_groups.append((x_test, y_test))
			group += 1

		except:
			print("failed to load group %d" % group)
			break
	return train_groups, test_groups



# Data i/o
# ========
#
# Returns dataset partitions (x_train, y_train), (x_test, y_test) by loading from
# the specified directory containing 'features.npy' and 'labels.npy'. Generates a
# new partition if none exists (as 'partition.npy'). If features and labels numpy
# arrays are not found in the provided directory, recursively searches sub-
# directories and attempts to concatenate features (by padding image widths to the
# width of the images in the largest dataset) and labels (assuming the number of
# labels is the same in all sub-directories).
def load_data(data_dir, test_ratio=0.1):
	# Try loading features from the provided directory
	try:
		x = np.load(os.path.join(data_dir, 'features.npy'))

		try:
			y = np.load(os.path.join(data_dir, 'labels.npy'))
		except:
			y = np.loadtxt(os.path.join(data_dir, 'labels.csv'), delimiter=',')  
		n = len(x)
		n_test = round(n * test_ratio)
		# Load an existing partition 
		try:
			p = np.load(os.path.join(data_dir, 'partition.npy'))
		# Or generate and save a new one
		except FileNotFoundError:
			p = np.random.permutation(n)
			np.save(os.path.join(data_dir, 'partition.npy'), p)
		# Return partitioned training and testing sets
		x_train = x[p[n_test:]]
		y_train = y[p[n_test:]]
		x_test = x[p[:n_test]]
		y_test = y[p[:n_test]]
		print('load_data(\'%s\')' % data_dir) 	# Print if successful

	# If the provided directory contains no features, check its sub-directories
	# recursively and concatenate any datasets found within
	except FileNotFoundError:
		print('Did not find dataset in \'%s\'. Searching subdirectories...' % data_dir)
		x_tr, y_tr, x_te, y_te = ([], [], [], [])
		for d in os.listdir(data_dir):
			try:
				(x_train, y_train), (x_test, y_test) = load_data(os.path.join(data_dir, d))
				x_tr.extend(x_train)
				y_tr.extend(y_train)
				x_te.extend(x_test)
				y_te.extend(y_test)
			except:
				pass
		# Standardize image widths to that of the widest dataset and return		
		try:		
			max_width = max([img.shape[1] for img in x_tr])
			n_train = len(x_tr)
			n_test = len(x_te)
			img_height = x_tr[0].shape[0]	
			x_train = image_list_to_np_array(x_tr, max_width)
			x_test = image_list_to_np_array(x_te, max_width)
			print('Concatenating datasets with standardized shapes:')
		except:
			print('Failed to find dataset(s) in subdirectories of \'%s\'' % data_dir)
			x_train, y_train, x_test, y_test = ([], [], [], [])
		# Labels arrays should have the same width or we shouldn't be concatenating
		# the datasets in these directories at all
		try:
			y_train = np.array(y_tr)
			y_test = np.array(y_te)
		except:
			print('Failed to concatenate labels arrays. Inconsistent dimensions.')
			y_train = []
			y_test = []
	# Print shapes and return
	train = (x_train, y_train)
	test = (x_test, y_test)
	print('training set: x.shape = ' + str(train[0].shape), '\ty.shape = ' + str(train[1].shape))
	print('    test set: x.shape = ' + str(test[0].shape), '\ty.shape = ' + str(test[1].shape))
	return train, test

