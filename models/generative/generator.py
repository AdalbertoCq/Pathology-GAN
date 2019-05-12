import tensorflow as tf
import math
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *


display = True


def mapping_network(z_input, z_dim, layers, spectral, activation, normalization, init='xavier', regularizer=None):
	if display:
		print('MAPPING NETWORK INFORMATION:')
		print('Layers:      ', layers)
		print('Normalization: ', normalization)
		print('Activation:    ', activation)
		print()

	with tf.variable_scope('mapping_network'):
		net = z_input
		for layer in range(layers):
			net = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=layer)	
		w_input = net

	return w_input


def generator_resnet(z_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, init='xavier', regularizer=None, cond_label=None, attention=None, up='upscale', bigGAN=False):
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
		if cond_label is not None: label = tf.concat([cond_label, label], axis=-1)

		# Dense.			
		net = dense(inputs=z_input_block, out_dim=1024, spectral=spectral, init=init, regularizer=regularizer, scope=1)			
		# net = batch_norm(inputs=net, training=is_train)
		# Introducing instance norm into dense layers.
		net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope='dense_1')
		net = activation(net)

		if bigGAN: label = z_splits[2]
		else: label = z_input
		if cond_label is not None: label = tf.concat([cond_label, label], axis=-1)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		# net = batch_norm(inputs=net, training=is_train)
		# Introducing instance norm into dense layers.
		net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope='dense_2')
		net = activation(net)
		
		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):

			if bigGAN: label = z_splits[3+layer] 
			else: label = z_input
			if cond_label is not None: label = tf.concat([cond_label, label], axis=-1)

			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral, init=init, regularizer=regularizer, 
								 activation=activation, normalization=normalization, cond_label=label)
			
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			net = normalization(inputs=net, training=is_train, c=label, spectral=spectral, scope=layer)
			net = activation(net)
			
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='logits')
		output = sigmoid(logits)
		
	print()
	return output


def generator_decoder_resnet(z_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, attention=None, up='upscale'):
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = list(reversed(channels[:layers]))

	if display:
		print('GENERATOR-DECODER INFORMATION:')
		print('Channels:      ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation:    ', activation)
		print('Attention H/W: ', attention)
		print()

	with tf.variable_scope('generator_decoder', reuse=reuse):
		# Doesn't work ReLU, tried.

		# Dense.			
		net = dense(inputs=z_input, out_dim=1024, spectral=spectral, scope=1)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		
		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		
		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):

			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral, activation=activation, normalization=normalization)
		
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, scope=layers)
		
			# if (vae_dim/2.) == net.shape.as_list()[1]:
			# 	lr_logs2_xi_z = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, scope='lr_logs2_xi_z')
		
			# if (vae_dim/2.) == net.shape.as_list()[1]:
			# 	scope = 'lr_mean_xi_z'
			# else:
			# 	scope = layer

			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, scope=layer)
			net = normalization(inputs=net, training=is_train)
			net = activation(net)

			# if vae_dim == net.shape.as_list()[1]:
			# 	lr_mean_xi_z = sigmoid(net)

			
		# Final outputs
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope='mean_xi_z')
		mean_xi_z = sigmoid(logits)

		# Final outputs
		logs2_xi_z = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope='logs2_xi_z')

		
	print()
	# return output, lr_mean_xi_z, lr_logs2_xi_z
	return mean_xi_z, logs2_xi_z


def generator_resnet_cond(z_input, c_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, up='upscale'):
	channels = [32, 64, 128, 256, 512, 1024]
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = list(reversed(channels[:layers]))

	if display:
		print('Generator Information.')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)

	with tf.variable_scope('generator', reuse=reuse):
		# Doesn't work ReLU, tried.
		# Z Input Shape = (None, 100)
		# C Input Shape = (None, 20)
		net = tf.concat([z_input, c_input], axis=1)

		# Dense.			
		net = dense(inputs=net, out_dim=1024, spectral=spectral, scope=1)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		
		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		
		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral,
								 activation=activation, normalization=normalization, c_input=c_input)
		
			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, scope=layer)
			net = normalization(inputs=net, training=is_train, c=c_input, spectral=spectral)
			net = activation(net)
		
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope='logits')
		output = sigmoid(logits)
		
	print()
	return output

def generator(z_input, image_channels, layers, spectral, activation, reuse, is_train, normalization):
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = list(reversed(channels[:layers]))

	if display:
		print('Generator Information.')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
	
	with tf.variable_scope('generator', reuse=reuse):
		# Doesn't work ReLU, tried.
		
		# Dense.
		net = dense(inputs=z_input, out_dim=1024, spectral=spectral, scope=1)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):
			# Conv.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type='transpose', spectral=spectral, scope=2*(layer+1)-1)
			net = normalization(inputs=net, training=is_train)
			net = activation(net)

			if layer != len(range(layers))-1:
				# Conv.
				net = convolutional(inputs=net, output_channels=reversed_channel[layer+1], filter_size=5, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=2*(layer+1))
				net = normalization(inputs=net, training=is_train)
				net = activation(net)

		# Conv.
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=2, stride=2, padding='SAME', conv_type='transpose', spectral=spectral, scope='logits')
		output = sigmoid(logits)
	
	print()
	return output
