import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *


def batch_norm(inputs, training, c=None, spectral=False, scope=False):
	output = tf.layers.batch_normalization(inputs=inputs, training=training)
	return output

def instance_norm(inputs, training, c=None, spectral=False, scope=False):
	# Not used: training
	output = tf.contrib.layers.instance_norm(inputs=inputs)
	return output

def layer_norm(inputs, training, c=None, spectral=False, scope=False):
	# Not used: training
	output = tf.contrib.layers.layer_norm(inputs=inputs, scope=False)
	return output

def group_norm(inputs, training, c=None, spectral=False, scope=False):
	# Not used: training
	output = tf.contrib.layers.group_norm(inputs=inputs)
	return output

def conditional_instance_norm(inputs, training, c, scope, spectral=False):
	input_dims = inputs.shape.as_list()
	if len(input_dims) == 4:
		batch, height, width, channels = input_dims
	else:
		batch, channels = input_dims

	with tf.variable_scope('conditional_instance_norm_%s' % scope):
		decay = 0.9
		epsilon = 1e-5

		# MLP for gamma, and beta.
		inter_dim = int((channels+c.shape.as_list()[-1])/2)
		net = dense(inputs=c, out_dim=inter_dim, scope=1, spectral=spectral, display=False)
		net = ReLU(net)
		gamma = dense(inputs=net, out_dim=channels, scope='gamma', spectral=spectral, display=False)
		gamma = ReLU(gamma)
		beta = dense(inputs=net, out_dim=channels, scope='beta', spectral=spectral, display=False)
		if  len(input_dims) == 4:
			gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
			beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
		
		if  len(input_dims) == 4:
			batch_mean, batch_variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
		else:
			batch_mean, batch_variance = tf.nn.moments(inputs, axes=[1], keep_dims=True)

		batch_norm_output = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, epsilon)

	return batch_norm_output

def conditional_batch_norm(inputs, training, c, scope, spectral=False):
	input_dims = inputs.shape.as_list()
	if len(input_dims) == 4:
		batch, height, width, channels = input_dims
	else:
		batch, channels = input_dims

	with tf.variable_scope('conditional_batch_norm_%s' % scope) :
		decay = 0.9
		epsilon = 1e-5

		test_mean = tf.get_variable("pop_mean", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
		test_variance = tf.get_variable("pop_var", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)

		# MLP for gamma, and beta.
		inter_dim = int((channels+c.shape.as_list()[-1])/2)
		net = dense(inputs=c, out_dim=inter_dim, scope=1, spectral=spectral, display=False)
		net = ReLU(net)
		gamma = dense(inputs=net, out_dim=channels, scope='gamma', spectral=spectral, display=False)
		gamma = ReLU(gamma)
		beta = dense(inputs=net, out_dim=channels, scope='beta', spectral=spectral, display=False)
		if  len(input_dims) == 4:
			gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
			beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)

		if training:
			if  len(input_dims) == 4:
				batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1, 2])
				# batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=True)
			else:
				batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1])
				# batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 1], keep_dims=True)
			ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
			ema_variance = tf.assign(test_variance, test_variance * decay + batch_variance * (1 - decay))
			with tf.control_dependencies([ema_mean, ema_variance]):
				batch_norm_output = tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, epsilon)
		else:
			batch_norm_output = tf.nn.batch_normalization(inputs, test_mean, test_variance, beta, gamma, epsilon)
	return batch_norm_output

