import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras import regularizers
from keras import losses
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.activations import sigmoid, tanh
from sklearn.decomposition import PCA

# Encoders
# ========
#
def build_encoder_lstm(input_shape, latent_size, lstm_sizes=(128,), dense_sizes=(64,), generative=True):

	# Input layer
	inputs = patch = Input(shape=input_shape, name='inputs')

	# Encoder LSTM layer(s)
	for idx, size in enumerate(lstm_sizes):
		patch = LSTM(size)(patch)
		patch = Dropout(0.2)(patch)
		if idx != len(lstm_sizes)-1:
			patch = RepeatVector(input_shape[0])(patch)

	# Encoder dense layer(s)
	for size in dense_sizes:
		patch = Dense(size, 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.01))(patch)

	# Latent space
	if generative:
		epsilon, latent = build_latent_generative(latent_size, inputs, patch)
		return Model(inputs=[inputs, epsilon], outputs=latent, name='encoder')
	else:
		latent = Dense(latent_size, name='latent')(patch)
		return Model(inputs, latent, name='encoder')

def build_encoder_cnn(input_shape, latent_size, dense_sizes=(64,), generative=True):

	# Input layer
	inputs = patch = Input(shape=input_shape, name='inputs')
	
	# Encoder convolutional and pooling layers
	patch = Conv2D(32, (128, 3), activation='relu', padding='valid')(patch)
	# patch = MaxPooling2D((2, 2), padding='same')(patch)
	patch = Conv2D(64, (1, 3), activation='relu', padding='valid')(patch)
	# patch = MaxPooling2D((2, 2), padding='same')(patch)
	# patch = Conv2D(64, (3, 3), activation='relu', padding='same')(patch)
	# patch = MaxPooling2D(2, padding='same')(patch)
	patch = Flatten()(patch)

	# Encoder dense layer(s)
	for size in dense_sizes:
		patch = Dense(size, 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.01))(patch)

	# Latent space
	if generative:
		epsilon, latent = build_latent_generative(latent_size, inputs, patch)
		return Model(inputs=[inputs, epsilon], outputs=latent, name='encoder')
	else:
		latent = Dense(latent_size, name='latent')(patch)
		return Model(inputs, latent, name='encoder')

def build_encoder_dnn(input_shape, latent_size, dense_sizes=(64,), generative=True):
	
	# Input layer
	inputs = patch = Input(shape=input_shape, name='inputs')
	patch = Flatten()(patch)

	# Encoder dense layer(s)
	for size in dense_sizes:
		patch = Dense(size, 
			activation='relu', 
			kernel_regularizer=regularizers.l2(0.01))(patch)

	# Latent space
	if generative:
		epsilon, latent = build_latent_generative(latent_size, inputs, patch)
		return Model(inputs=[inputs, epsilon], outputs=latent, name='encoder')
	else:
		latent = Dense(latent_size, name='latent')(patch)
		return Model(inputs, latent, name='encoder')

def build_latent_generative(latent_size, inputs, latent_inputs):

	# Parameterized latent space
	latent_mean = Dense(latent_size, name='latent_mean')(latent_inputs)
	latent_log_var = Dense(latent_size, name='latent_log_var')(latent_inputs)

	# Pass latent parameters through a transparent layer that comptues and adds the
	# KL-divergence to the network loss
	latent_mean, latent_log_var = KLDivergenceLayer()([latent_mean, latent_log_var])
	latent_std = Lambda(lambda t: K.exp(.5*t), name='latent_std')(latent_log_var)

	# Sampling (use random_normal tensor so we don't have to feed another input into
	# the model when calling .fit() or .predict())
	epsilon = Input(tensor=K.random_normal(shape=(K.shape(inputs)[0], latent_size)),
		name = 'epsilon')
	latent_epsilon = Multiply()([latent_std, epsilon])
	latent = Add()([latent_mean, latent_epsilon])

	return (epsilon, latent)

class KLDivergenceLayer(Layer):
	""" Identity transform layer that adds KL divergence
    to the final model loss.
    """
	def __init__(self, *args, **kwargs):
		self.is_placeholder = True
		super(KLDivergenceLayer, self).__init__(*args, **kwargs)

	def call(self, inputs):
		mu, log_var = inputs
		kl_batch = - .5 * K.sum(1 + log_var -
		                        K.square(mu) -
		                        K.exp(log_var), axis=-1)
		self.add_loss(K.mean(kl_batch), inputs=inputs)
		return inputs

# Regressor
# =========
#
def build_regressor(latent_size, output_size, dense_sizes=()):
	# Decoder input
	inputs = patch = Input(shape=(latent_size,), name='latent_inputs')
	# Decoder dense layer(s)
	for size in dense_sizes:
		patch = Dense(size, 
			activation=None,
			kernel_regularizer=regularizers.l2(0.01))(patch)
	# Output layer
	patch = Dense(output_size, activation='sigmoid', name='outputs')(patch)
	# Output scaling
	outputs = Lambda(lambda x: x * 127.0, name='output_scaling')(patch)
	return Model(inputs, outputs, name='regressor')

# End-to-end Model
# ================
#
# Assemble end-to-end model from encoder and regressor, compile, and test
def build_end_to_end(encoder, regressor):
	# End-to-end model 
	model = Model(encoder.inputs, regressor(encoder(encoder.inputs)), name='model')
	model.compile(loss='mse', optimizer='rmsprop')
	# Print summaries
	encoder.summary()
	regressor.summary()
	model.summary()
	# If the model has no specified sequence length, use 256
	test_shape = encoder.layers[0].batch_input_shape[1:]
	if test_shape[0] == None:
		test_shape = (256, test_shape[1])
	# Verify correct encoder/regressor configuration with some synthetic data
	x = np.random.normal(0, 1, (64,) + test_shape)
	p = model.predict(x, batch_size=len(x))
	z = encoder.predict(x, batch_size=len(x))
	p_h = regressor.predict(z, batch_size=len(z))
	# Check
	if sum(sum(p == p_h)) == p.size:
		print("Incorrect encoder/regressor configuration")
	# assert sum(sum(p == p_h)) == p.size, "Incorrect encoder/regressor configuration"
	return model

# Evaluation
# ==========
# 
def model_eval(model, x_test, y_test, model_dir=None, file_suffix=None):
	# Eval
	score = model.evaluate(x_test, y_test, batch_size=len(x_test))
	print('Score: %f' % score)
	# Predictions
	y_hat = model.predict(x_test, batch_size=len(x_test), verbose=1, steps=None)
	print(np.concatenate((y_test, np.round(y_hat)), axis=1))
	# Export error distribution plots if a directory is provided
	if model_dir is not None:
		error_plots(y_test, np.round(y_hat), model_dir, file_suffix=file_suffix)

def model_eval_varlen(model, x_test, y_test, model_dir=None, file_suffix=None):
	# Eval
	score = model.evaluate(x_test, y_test, batch_size=1)
	print('Score: %f' % score)
	# Predictions
	y_hat = np.zeros(y_test.shape)
	for i, x in enumerate(x_test):
		y_hat[i,:] = model.predict(x.reshape((1,) + x.shape), batch_size=1, verbose=1, steps=None)
	return score, y_hat

# Produce matrix of error distribution plots for each target
def error_plots(y, y_hat, path, file_suffix=None):
	n, m = y.shape
	for i in range(m):
		bin, err, err_abs = errs(y, y_hat, i)	# Error dist. over i^th variable values
		try:
			w = np.diff(bin)[0] / 2 				# Plot's bar width
		except:
			w = np.diff(bin) / 2
		min_err = np.min(err)
		max_err = np.max(err_abs)
		for j in range(m):
			ax = plt.subplot(m, m, i*m + j+1)
			b1 = ax.bar(bin-w/2, err[:, j], w, color='r')
			b2 = ax.bar(bin+w/2, err_abs[:, j], w, color='b')
			ax.set_ylim(bottom=min_err, top=max_err)
			ax.set_xlabel('Value P%d' % i)	
			ax.set_ylabel('Test Error P%d' % j)
			ax.legend((b1, b2), ('error', 'abs_error'))
	plt.ioff()
	fig = plt.gcf()
	dpi = 110
	fig.set_size_inches((1440/dpi, 900/dpi))
	fname = 'err_dist'
	if file_suffix is not None:
		fname += file_suffix
	plt.savefig(os.path.join(path, fname + '.png'), bbox_inches='tight', dpi=dpi)
	plt.close(fig)

# Helper function for err_plots(); computes a single error distribution
def errs(y, y_hat, idx_indep):
	n, m = y.shape 						# n examples, m parameters
	y_indep = y[:, idx_indep]	 		# Indep. variable for err distribution
	bin = np.unique(y_indep)			# Distribution bins
	err = np.zeros((len(bin), m)) 		# Error (signed)
	err_abs = np.zeros((len(bin), m))	# Absolute error
	for i in range(n):			
		# Distribution bin for this value of indep. variable
		b = np.where(bin == y_indep[i])[0][0]
		for j in range(m):
			# Err and abs. err
			e = y[i, j] - y_hat[i, j]
			err[b][j] += e
			err_abs[b][j] += abs(e)
	return bin, err, err_abs


# PCA and plotting
# ================
# 
def pca(latent):
	pca = PCA(n_components=latent.shape[1])
	latent_projected = pca.fit_transform(latent)
	weights = pca.components_.T
	biases = pca.mean_
	return weights, biases, latent_projected

def scatter_latent(z, z_projected=None):
	fig = plt.figure()
	if z_projected is not None:
		ax1 = fig.add_subplot(121, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')
		ax1.scatter(z[:,0], z[:,1], z[:,2])
		ax2.scatter(z_projected[:,0], z_projected[:,1], z_projected[:,2])
	else:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(z[:,0], z[:,1], z[:,2])
	plt.show()

# Exporting Keras models
# ======================
# 
def export_keras(model_dir, model, encoder, regressor):
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	# Save models to JSON 
	json = encoder.to_json()
	with open(os.path.join(model_dir, 'encoder.json'), 'w') as jf:
		jf.write(json)
	json = regressor.to_json()
	with open(os.path.join(model_dir, 'regressor.json'), 'w') as jf:
		jf.write(json)
	json = model.to_json()
	with open(os.path.join(model_dir, 'model.json'), 'w') as jf:
		jf.write(json)
	# Save weights to HDF5	
	encoder.save_weights(os.path.join(model_dir, 'encoder.h5'))
	regressor.save_weights(os.path.join(model_dir, 'regressor.h5'))
	model.save_weights(os.path.join(model_dir, 'model.h5'))

# Exporting to Max/MSP 
# ====================
# Exports regressor, pca, and latent space scaling parameters to plain text in a set
# of directories parsed by the Max external 'timbremap' and its C classes 'dense_layer'
# 'pca_layer', and 'vec_scale'. 
#
# Export latent space statistics
def export_vec_scale(out_dir, latent):
	out_dir = os.path.join(out_dir, 'vec_scale')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	# Export min, range, mean, and std. dev. of test data projected into latent space
	export_matrix(os.path.join(out_dir, 'min'), latent.min(0))
	export_matrix(os.path.join(out_dir, 'range'), latent.max(0) - latent.min(0))
	export_matrix(os.path.join(out_dir, 'mean'), latent.mean(0))
	export_matrix(os.path.join(out_dir, 'std'), latent.std(0))

def export_pca_layer(out_dir, weights, biases):
	out_dir = os.path.join(out_dir, 'pca_layer')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	export_layer(out_dir, weights, biases)

# Export regressor layers
def export_regressor(out_dir, regressor):
	for idx, layer in enumerate(regressor.layers[1:]):
		if layer.get_weights():
			export_layer(os.path.join(out_dir, 'dense_layer_%d' % idx),
					layer.get_weights()[0],
					layer.get_weights()[1],
					layer.activation)

# Export a dense_layer 
def export_layer(out_dir, weights, biases, activation_func=None):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	try:	# Inverse 
		weights_inv = np.linalg.inv(weights)	
	except:	# Pseudoinverse
		weights_inv = np.linalg.pinv(weights)	
	# Export weights and biases
	export_matrix(os.path.join(out_dir, 'weights'), weights)
	export_matrix(os.path.join(out_dir, 'biases'), biases)
	export_matrix(os.path.join(out_dir, 'weights_inv'), weights_inv)
	# Export Activation
	if activation_func == sigmoid:		
		act_str = 'sigmoid'
	elif activation_func == tanh:
		act_str = 'tanh'
	else:
		return
	fh = open(os.path.join(out_dir, 'activation'), 'w')
	fh.write(act_str)
	fh.write('\n')

# Write flattened matrix values
def export_matrix(f_path, np_arr):
	fh = open(f_path, 'w')
	for val in np_arr.flatten():
		fh.write('%.32f\n' % val)
	fh.close()
	np.save(f_path, np_arr)