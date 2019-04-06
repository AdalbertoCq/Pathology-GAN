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

	batch, height, width, channels = inputs.shape.as_list()

	with tf.variable_scope('conditional_instance_norm_%s' % scope):

		# MLP for gamma, and beta.
		inter_dim = int((channels+c.shape.as_list()[-1])/2)
		net = dense(inputs=c, out_dim=inter_dim, scope=1, spectral=spectral, display=False)
		net = ReLU(net)
		gamma = dense(inputs=net, out_dim=channels, scope='gamma', spectral=spectral, display=False)
		gamma = ReLU(gamma)
		beta = dense(inputs=net, out_dim=channels, scope='beta', spectral=spectral, display=False)

		gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
		beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
		
		mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
		print('Gamma', gamma.shape)
		print('Beta', beta.shape)
		print('Mean', mean.shape)
		print('Var', variance.shape)
		batch_norm_output = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-10)

	return batch_norm_output