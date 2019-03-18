import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *


def batch_norm(inputs, training):
	output = tf.layers.batch_normalization(inputs=inputs, training=training)
	return output

def instance_norm(inputs, training):
	# Not used: training
	output = tf.contrib.layers.instance_norm(inputs=inputs)
	return output

def layer_norm(inputs, training):
	# Not used: training
	output = tf.contrib.layers.layer_norm(inputs=inputs)
	return output

def group_norm(inputs, training):
	# Not used: training
	output = tf.contrib.layers.group_norm(inputs=inputs)
	return output

def conditional_instance_norm(inputs, training, c=None, spectral=False):

	batch, height, width, channels = inputs.shape.as_list()

	# MLP for gamma, and beta.
	net = dense(inputs, (channels+c)/2, scope, use_bias=True, spectral=spectral)
	net = ReLU(net)
	gamma = dense(inputs, channels, scope, use_bias=True, spectral=spectral)
	gamma = ReLU(beta)
	beta = dense(inputs, channels, scope, use_bias=True, spectral=spectral)

	mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
	x = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-10)

	return batch_norm_output