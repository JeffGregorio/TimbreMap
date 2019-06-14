import os
import numpy as np

# Verify invertibility of mapping layers using Python prototypes of the Max/MSP
# runtime external
def test_max(model_dir, scale_mode='uniform'):

	# Parameters for mapping control space to latent space
	if scale_mode == 'uniform':
		scale_layer = MaxVecScale(os.path.join(model_dir, 'vec_scale'))
	elif scale_mode == 'normal':
		scale_layer = MaxGaussianScale(os.path.join(model_dir, 'vec_scale'))

	# PCA layer
	try:
		pca_layer = MaxPCALayer(os.path.join(model_dir, 'pca_layer'))
		use_pca = True
	except:
		use_pca = False

	# Dense layer(s)
	dense_layers = []
	layer_idx = 0
	layer_dir = os.path.join(model_dir, 'dense_layer_%d' % layer_idx)
	while os.path.exists(layer_dir):
		dense_layers.append(MaxDenseLayer(layer_dir))
		layer_idx += 1
		layer_dir = os.path.join(model_dir, 'dense_layer_%d' % layer_idx)

	# Test data
	rng = np.linspace(0.01, 0.99, 10)
	c = np.array([[i, j, k] for i in rng for j in rng for k in rng])

	# Forward pass
	patch = scale_layer.process_forward(c)
	if use_pca:
		patch = pca_layer.process_forward(patch)
	for layer in dense_layers:
		patch = layer.process_forward(patch)

	# Backward pass
	for layer in reversed(dense_layers):
		patch = layer.process_backward(patch)
	if use_pca:
		patch = pca_layer.process_backward(patch)
	c_hat = scale_layer.process_backward(patch)

	# Return round-trip error per example
	return np.sum(np.abs(c - c_hat)) / len(c)

# Max/MSP external tests/prototypes
# =================================
#
class MaxLayer:
	def __init__(self, layer_dir):
		self._w = np.load(os.path.join(layer_dir, 'weights.npy'))
		self._wi = np.load(os.path.join(layer_dir, 'weights_inv.npy'))
		self._b = np.load(os.path.join(layer_dir, 'biases.npy'))
		return

class MaxDenseLayer(MaxLayer):
	def __init__(self, layer_dir):
		MaxLayer.__init__(self, layer_dir)
		try:
			f = open(os.path.join(layer_dir, 'activation'))
			lines = [line.rstrip('\n') for line in f]
			f.close()
		except:
			lines = ['linear']
		if lines[0] == 'linear':
			self._act = lambda x: x
			self._act_inv = lambda x: x
		elif lines[0] == 'sigmoid':
			self._act = lambda x: (np.exp(x) / (np.exp(x) + 1)) * 127.0
			self._act_inv = lambda x: np.log((x/127.0) / (1 - (x/127.0) + 0.0000000001))
		elif lines[0] == 'tanh':
			self._act = lambda x: np.tanh(x)
			self._act_inv = lambda x: np.atanh(x)
		elif lines[0] == 'leakyrelu':		# TO DO: actually implement this
			self._act = lambda x: x
			self._act_inv = lambda x: x
	def process_forward(self, inputs):
		return self._act(np.dot(inputs, self._w) + self._b)
	def process_backward(self, inputs):
		return np.dot(self._act_inv(inputs) - self._b, self._wi)

class MaxPCALayer(MaxLayer):
	def __init__(self, layer_dir):
		MaxLayer.__init__(self, layer_dir)
		return 
	def process_forward(self, inputs):
		return np.dot(inputs, self._wi) + self._b
	def process_backward(self, inputs):
		return np.dot(inputs - self._b, self._w)

class MaxVecScale:
	def __init__(self, rescale_dir):
		self._bias = np.load(os.path.join(rescale_dir, 'min.npy'))
		self._scale = np.load(os.path.join(rescale_dir, 'range.npy'))
		return
	def process_forward(self, inputs):
		return inputs * self._scale + self._bias
	def process_backward(self, inputs):
		return (inputs - self._bias) / self._scale

class MaxGaussianScale:
	def __init__(self, rescale_dir):
		self._mean = np.load(os.path.join(rescale_dir, 'mean.npy'))
		self._std = np.load(os.path.join(rescale_dir, 'std.npy'))
		return
	def process_forward(self, inputs):
		# return norm.ppf(inputs, loc=self._mean, scale=self._std)
		return norm_ppf(inputs, self._mean, self._std)
	def process_backward(self, inputs):
		# return norm.cdf(inputs, loc=self._mean, scale=self._std)
		return norm_cdf(inputs, self._mean, self._std)

def norm_cdf(x, mu=0.0, sig=1.0):
	return 0.5 * (1 + erf((x - mu) / (sig*2**0.5)))

def norm_ppf(x, mu=0.0, sig=1.0):
	return mu + sig*2**0.5 * erfi(2*x - 1)

def erf(x):
	y = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			y[i, j] = util_erf(x[i, j])
	return y

def erfi(x):
	y = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			y[i, j] = util_erf_inv(x[i, j])
	return y

# Forward and inverse error functions taken from:
# https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
# which was taken from a 2008 paper
# http://www.academia.edu/9730974/A_handy_approximation_for_the_error_function_and_its_inverse
def util_erf(x):
	sgn = -1.0 if x < 0 else 1.0
	xx = x * x
	axx = 0.147 * xx
	return sgn * (1 - np.exp(-xx * (4/np.pi + axx) / (1 + axx))) ** 0.5

def util_erf_inv(x):
	sgn = -1.0 if x < 0 else 1.0
	lnx = np.log((1 - x) * (1 + x))
	tt1 = 2 / (np.pi * 0.147) + 0.5 * lnx
	tt2 = 1 / (0.147) * lnx
	return sgn * np.sqrt((-tt1 + np.sqrt(tt1 * tt1 - tt2)))



