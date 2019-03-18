import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *

display = True

def discriminator_resnet(images, layers, spectral, activation, reuse, normalization=None, attention=None, down='downscale'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]

	if display:
		print('DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope('discriminator', reuse=reuse):
		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, activation=activation)
			if display: 
				print('ResBlock Layer: channels %4s filter_size=3, stride=1, padding=SAME, conv_type=convolutional scope=%s Output Shape: %s' % (channels[layer], layer, net.shape))

			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, scope=layers)
				print('Att. Layer    : channels %4s' % channels[layer])

			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			if display: 
				print('Conv Layer:     channels %4s filter_size=4, stride=2, padding=SAME, conv_type=%s scope=%s Output Shape: %s' % (channels[layer], down, layer, net.shape))

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)
		if display: 
			print('Dense Layer:    dim.     %4s Output Shape: %s' % (channels[-1], net.shape))

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, scope=2)				
		output = sigmoid(logits)
		if display: 
			print('Dense Layer:    dim.       1 Output Shape: %s' % net.shape)
			
	print()
	return output, logits


def discriminator(images, layers, spectral, activation, reuse, normalization=None):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	
	if display:
		print('Discriminator Information.')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
	with tf.variable_scope('discriminator', reuse=reuse):
		# Padding = 'Same' -> H_new = H_old // Stride

		for layer in range(layers):
			# Down.
			if display: print('Conv Layer: channels %s filter_size=5, stride=2, padding=SAME, conv_type=transpose scope=%s' % (channels[layer], layer))
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=5, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer+1)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)

		# Flatten.
		net = tf.layers.flatten(inputs=net)
		
		# Dense.
		if display: print('Dense Layer: Dim=%s' % channels[-1])
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)
		
		# Dense
		if display: print('Dense Layer: Dim=1')
		logits = dense(inputs=net, out_dim=1, spectral=spectral, scope=2)				
		output = sigmoid(logits)

	print()
	return output, logits