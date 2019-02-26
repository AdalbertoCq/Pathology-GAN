import tensorflow as tf


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