import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *


def discriminator_resnet(images, layers, use_bn, spectral, activation, reuse):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]

	with tf.variable_scope('discriminator', reuse=reuse):
		for layer in range(layers):
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', channels=channels[layer], scope=layer, is_training=True, use_bn=use_bn, use_bias=True, 
								 spectral=spectral, activation=activation)
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=5, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer)
			if use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = activation(net)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, scope=1)				
		if use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
		net = activation(net)
		# Shape = (None, 1024)

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, scope=2)				
		# Shape = (None, 1)
		output = signmoid(logits)

	return output, logits


def discriminator_cnn(images, layers, use_bn, spectral, activation, reuse):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	
	print('Discriminator.')
	print('Channels: ', channels[:layers])
	with tf.variable_scope('discriminator', reuse=reuse):
		# Padding = 'Same' -> H_new = H_old // Stride

		# Input Shape = (None, 224, 224, 3)
		for layer in range(layers):
			# Conv.
			print('Conv Layer: channels %s ilter_size=5, stride=2, padding=SAME, conv_type=transpose scope=%s' % (channels[layer], layer))
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=5, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer)
			# net = attention_block(x=net, i=1)
			if use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = activation(net)
			# Shape = (None, 112, 112, 32)

		# Flatten.
		net = tf.layers.flatten(inputs=net)
		# Shape = (None, 7*7*512)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, scope=1)				
		if use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
		net = activation(net)
		# Shape = (None, 1024)

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, scope=2)				
		# Shape = (None, 1)
		output = tf.nn.sigmoid(x=logits)

		# Padding = 'Same' -> H_new = H_old // Stride
	print()
	return output, logits