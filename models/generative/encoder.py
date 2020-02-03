import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *

display = True

def encoder_resnet(images, layers, z_dim, spectral, activation, reuse, is_train, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope('encoder', reuse=reuse):
		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		net = residual_block_dense(inputs=net, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=1)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-2], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		net = residual_block_dense(inputs=net, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=2)

		# Dense
		# mus = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		# sigmas = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=4)	
		# sigmas= ReLU(sigmas) + 1e-12
		# sigmas= tf.nn.softplus(sigmas)

		z_hat = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)		

	print()
	# return mus, sigmas
	return z_hat