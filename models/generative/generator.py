import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *

def generator_resnet(z_input, image_channels, layers, spectral, activation, reuse, is_train):
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = reversed(channels[:layers])

	with tf.variable_scope('generator', reuse=reuse):
		# Doesn't work ReLU, tried.
		# Input Shape = (None, 100)
		# Dense.			
		net = dense(inputs=z_input, out_dim=1024, spectral=spectral, scope=1)				
		net = tf.layers.batch_normalization(inputs=net, training=is_train)
		net = activation(net)
		# Shape = (None, 1024)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = tf.layers.batch_normalization(inputs=net, training=is_train)
		net = activation(net)
		# Shape = (None, 256*7*7)

		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')
		# Shape = (None, 7, 7, 256)

		for layer in range(layers):
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', channels=reversed_channel[layer], scope=layer, is_training=is_train, use_bn=True,
								 spectral=spectral, activation=activation)

			# Conv.
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=5, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=2*(layers+1))
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = activation(net)

		net = convolutional(inputs=net, output_channels=image_channels, filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=2*(layers+1))
		net = sigmoid(net)


def generator(z_input, image_channels, layers, spectral, activation, reuse, is_train):
	channels = [32, 64, 128, 256, 512, 1024]
	reversed_channel = list(reversed(channels[:layers]))

	print('Generator.')
	print('Channels: ', channels[:layers])
	
	with tf.variable_scope('generator', reuse=reuse):
		# Doesn't work ReLU, tried.
		# Input Shape = (None, 100)
		# Dense.			
		net = dense(inputs=z_input, out_dim=1024, spectral=spectral, scope=1)				
		net = tf.layers.batch_normalization(inputs=net, training=is_train)
		net = activation(net)
		# Shape = (None, 1024)

		# Dense.
		net = dense(inputs=net, out_dim=256*7*7, spectral=spectral, scope=2)				
		net = tf.layers.batch_normalization(inputs=net, training=is_train)
		net = activation(net)
		# Shape = (None, 256*7*7)

		# Reshape
		net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')
		# Shape = (None, 7, 7, 256)

		for layer in range(layers):
			# Conv.
			print('Conv Layer: channels %s ilter_size=2, stride=2, padding=SAME, conv_type=transpose scope=%s' % (reversed_channel[layer], 2*(layer+1)-1))
			net = convolutional(inputs=net, output_channels=reversed_channel[layer], filter_size=2, stride=2, padding='SAME', conv_type='transpose', spectral=spectral, scope=2*(layer+1)-1)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = activation(net)
			# Shape = (None, 14, 14, 256)

			if layer != len(range(layers))-1:
				# Conv.
				print('Conv Layer: channels %s ilter_size=5, stride=1, padding=SAME, conv_type=convolutional scope=%s' % (reversed_channel[layer+1], 2*(layer+1)))
				net = convolutional(inputs=net, output_channels=reversed_channel[layer+1], filter_size=5, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, scope=2*(layer+1))
				net = tf.layers.batch_normalization(inputs=net, training=is_train)
				net = activation(net)

		# Conv.
		print('Logits Layer: channels %s ilter_size=5, stride=1, padding=SAME, conv_type=convolutional scope=%s' % (image_channels, 2*(layer+1)))
		logits = convolutional(inputs=net, output_channels=image_channels, filter_size=2, stride=2, padding='SAME', conv_type='transpose', spectral=spectral, scope=2*(layer+1))
		# Shape = (None, 448, 448, 3)
		output = tf.nn.sigmoid(x=logits, name='output')
	
	print()
	return output


	