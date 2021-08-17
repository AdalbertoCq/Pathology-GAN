import tensorflow as tf
import math
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *


display = True


def mapping_resnet(z_input, z_dim, layers, reuse, is_train, spectral, activation, normalization, init='xavier', regularizer=None):
	if display:
		print('MAPPING NETWORK INFORMATION:')
		print('Layers:      ', layers)
		print('Normalization: ', normalization)
		print('Activation:    ', activation)
		print()

	with tf.variable_scope('mapping_network', reuse=reuse):
		net = z_input
		for layer in range(layers):
			net = residual_block_dense(inputs=net, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=layer)
		z_map = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=1)
	print()

	return z_map


def generator_resnet(z_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, init='xavier', noise_input_f=False, regularizer=None, cond_label=None, attention=None, up='upscale', bigGAN=False):
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = list(reversed(channels[:layers]))

	# Question here: combine z dims for upscale and the conv after, or make them independent.
	if bigGAN:
		z_dim = z_input.shape.as_list()[-1]
		blocks = 2 + layers
		block_dims = math.floor(z_dim/blocks)
		remainder = z_dim - block_dims*blocks
		if remainder == 0:
			z_sets = [block_dims]*(blocks + 1)
		else:
			z_sets = [block_dims]*blocks + [remainder]
		z_splits = tf.split(z_input, num_or_size_splits=z_sets, axis=-1)


	if display:
		print('GENERATOR INFORMATION:')
		print('Channels:      ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation:    ', activation)
		print('Attention H/W: ', attention)
		print()

	with tf.variable_scope('generator', reuse=reuse):
		if bigGAN: 
			z_input_block = z_splits[0]
			label = z_splits[1]
		else:
			z_input_block = z_input
			label = z_input
		if cond_label is not None: 
			if 'training_gate' in cond_label.name:
				label = cond_label
			else:
				label = tf.concat([cond_label, label], axis=-1)

		# Dense.			
		net = dense(inputs=z_input_block, out_dim=1024, spectral=spectral, init=init, regularizer=regularizer, scope=1)			
		net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope='dense_1')
		net = activation(net)

		if bigGAN: label = z_splits[2]
		else: label = z_input
		if cond_label is not None: 
			if 'training_gate' in cond_label.name:
				label = cond_label
			else:
				label = tf.concat([cond_label, label], axis=-1)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope='dense_2')
		net = activation(net)
		
		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):

			if bigGAN: label = z_splits[3+layer] 
			else: label = z_input
			if cond_label is not None: 
				if 'training_gate' in cond_label.name:
					label = cond_label
				else:
					label = tf.concat([cond_label, label], axis=-1)

			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral, init=init, regularizer=regularizer, noise_input_f=noise_input_f, 
								 activation=activation, normalization=normalization, cond_label=label)
			
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
				# ResBlock.
				# net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer+layers, is_training=is_train, spectral=spectral, init=init, regularizer=regularizer, noise_input_f=noise_input_f, 
									 # activation=activation, normalization=normalization, cond_label=label)
			
			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if noise_input_f:
				net = noise_input(inputs=net, scope=layer)
			# net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope=layer+layers)
			net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope=layer)
			net = activation(net)
			
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='logits')
		output = sigmoid(logits)
		
	print()
	return output


def generator_resnet_style(z_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, init='xavier', noise_input_f=False, regularizer=None, cond_label=None, attention=None, up='upscale'):
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = list(reversed(channels[:layers]))

	if display:
		print('GENERATOR INFORMATION:')
		print('Channels:      ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation:    ', activation)
		print('Attention H/W: ', attention)
		print()

	with tf.variable_scope('generator', reuse=reuse):

		z_input_block = z_input[:, :, 0]
		label = z_input[:, :, 0]

		# Dense.			
		net = dense(inputs=z_input_block, out_dim=1024, spectral=spectral, init=init, regularizer=regularizer, scope=1)			
		net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope='dense_1')
		net = activation(net)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope='dense_2')
		net = activation(net)
		
		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):

			label = z_input[:, :, layer]
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral, init=init, regularizer=regularizer, noise_input_f=noise_input_f, 
								 activation=activation, normalization=normalization, cond_label=label)
			print('Z input:', layer)

			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
				# ResBlock.
				# net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer+layers, is_training=is_train, spectral=spectral, init=init, regularizer=regularizer, noise_input_f=noise_input_f, activation=activation, normalization=normalization, cond_label=label)
				
			label = z_input[:, :, layer+1]
			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if noise_input_f:
				net = noise_input(inputs=net, scope=layer)
			net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope=layer)
			net = activation(net)
			print('Z input:', layer+1)
			
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='logits')
		output = sigmoid(logits)
		
	print()
	return output
