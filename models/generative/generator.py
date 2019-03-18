import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *

display = True

def generator_resnet(z_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, attention=None, up='upscale'):
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
		# Doesn't work ReLU, tried.

		# Dense.			
		net = dense(inputs=z_input, out_dim=1024, spectral=spectral, scope=1)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		if display: 
			print('Dense Layer:    dim.    1024 Output Shape: %s' % net.shape)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		if display: 
			print('Dense Layer:    dim. 256*7*7 Output Shape: %s' % net.shape)

		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral, activation=activation, normalization=normalization)
			if display:
				print('ResBlock Layer: channels %4s filter_size=3, stride=1, padding=SAME, conv_type=convolutional scope=%s Output Shape: %s' % (reversed_channel[layer], layer, net.shape))

			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, scope=layers)
				print('Att. Layer    : channels %4s' % reversed_channel[layer])

			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, scope=layer)
			net = normalization(inputs=net, training=is_train)
			net = activation(net)
			if display:
				print('Conv Layer:     channels %4s filter_size=2,  stride=2, padding=SAME, conv_type=%s scope=%s Output Shape: %s' % (reversed_channel[layer], up, layer, net.shape))

		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer+1)
		output = sigmoid(logits)
		if display: 
			print('Logits Layer:   channels %4s filter_size=3, stride=1, padding=SAME, conv_type=convolutional scope=%s Output Shape: %s' % (image_channels, layer+1, net.shape))

	print()
	return output

def generator_resnet_cond(z_input, c_input, image_channels, layers, spectral, activation, reuse, is_train, normalization, up='upscale'):
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
		if display: 
			print('Dense Layer:    dim.    1024 Output Shape: %s' % net.shape)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)
		if display: 
			print('Dense Layer:    dim. 256*7*7 Output Shape: %s' % net.shape)

		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, spectral=spectral,
								 activation=activation, normalization=normalization, c_input=c_input)
			if display:
				print('ResBlock Layer: channels %4s filter_size=3, stride=1, padding=SAME, conv_type=convolutional scope=%s Output Shape: %s' % (reversed_channel[layer], layer, net.shape))

			# Up.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type=up, spectral=spectral, scope=layer)
			net = normalization(inputs=net, training=is_train, c=c_input, spectral=spectral)
			net = activation(net)
			if display:
				print('Conv Layer:     channels %4s ilter_size=2,  stride=2, padding=SAME, conv_type=%s scope=%s Output Shape: %s' % (reversed_channel[layer], up, layer, net.shape))

		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer+1)
		output = sigmoid(logits)
		if display: 
			print('Logits Layer:   channels %4s filter_size=3, stride=1, padding=SAME, conv_type=convolutional scope=%s Output Shape: %s' % (image_channels, layer+1, net.shape))

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
		if display: print('Dense Layer: Dim=1024')
		net = dense(inputs=z_input, out_dim=1024, spectral=spectral, scope=1)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Dense.
		if display: print('Dense Layer: Dim=256*7*7')
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')

		for layer in range(layers):
			# Conv.
			if display: print('Conv Layer: channels %s filter_size=2, stride=2, padding=SAME, conv_type=transpose scope=%s' % (reversed_channel[layer], 2*(layer+1)-1))
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type='transpose', spectral=spectral, scope=2*(layer+1)-1)
			net = normalization(inputs=net, training=is_train)
			net = activation(net)

			if layer != len(range(layers))-1:
				# Conv.
				if display: print('Conv Layer: channels %s filter_size=5, stride=1, padding=SAME, conv_type=convolutional scope=%s' % (reversed_channel[layer+1], 2*(layer+1)))
				net = convolutional(inputs=net, output_channels=reversed_channel[layer+1], filter_size=5, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=2*(layer+1))
				net = normalization(inputs=net, training=is_train)
				net = activation(net)

		# Conv.
		if display: print('Logits Layer: channels %s filter_size=2, stride=2, padding=SAME, conv_type=convolutional scope=%s' % (image_channels, 2*(layer+1)))
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=2, stride=2, padding='SAME', conv_type='transpose', spectral=spectral, scope=2*(layer+1))
		output = sigmoid(logits)
	
	print()
	return output
