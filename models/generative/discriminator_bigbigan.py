import tensorflow as tf
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *

display = True

def discriminator_resnet(images, encoding, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, label_t='cat', infoGAN=False, c_dim=None):
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

		# Discriminator with conditional projection.
		if label is not None:
			batch_size, label_dim = label.shape.as_list()
			embedding_size = channels[-1]
			# Categorical Embedding.
			print(label_t)
			if label_t == 'cat':
				# emb = embedding(shape=(label_dim, embedding_size), init=init, regularizer=regularizer, power_iterations=1)
				emb = embedding(shape=(label_dim, embedding_size), init=init, power_iterations=1)
				index = tf.argmax(label, axis=-1)
				label_emb = tf.nn.embedding_lookup(emb, index)
			# Linear conditioning, using NN to produce embedding.
			else:
				inter_dim = int((label_dim+net.shape.as_list()[-1])/2)
				net_label = dense(inputs=label, out_dim=inter_dim, spectral=spectral, init='xavier', regularizer=None, scope='label_nn_1')
				if normalization is not None: net_label = normalization(inputs=net_label, training=True)
				net_label = activation(net_label)
				label_emb = dense(inputs=net_label, out_dim=embedding_size, spectral=spectral, init='xavier', regularizer=None, scope='label_nn_2')

		with tf.variable_scope('discriminator_F', reuse=reuse):
			if display: print('discriminator_F:')
			for layer in range(layers):
				# ResBlock.
				net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
									 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
				# Attention layer. 
				if attention is not None and net.shape.as_list()[1]==attention:
					net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
				
				# Down.
				net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
				if normalization is not None: net = normalization(inputs=net, training=True)
				net = activation(net)
				
			# Flatten.
			net = tf.layers.flatten(inputs=net)

			# Dense.
			F_s_x = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
			if normalization is not None: net = normalization(inputs=net, training=True)
			F_s_x = activation(F_s_x)

			# Dense
			s_x = dense(inputs=F_s_x, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
			output = sigmoid(s_x)

		# H: MLP for encodings, 8 layers or 4 ResNet and 2048 dim according to BigBiGAN.
		with tf.variable_scope('discriminator_H', reuse=reuse):
			if display: print('Discriminator_H:')
			H_s_z = dense(inputs=encoding, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)
			H_s_z = activation(H_s_z)
			H_s_z = residual_block_dense(inputs=H_s_z, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=1)
			H_s_z = residual_block_dense(inputs=H_s_z, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=2)
			H_s_z = residual_block_dense(inputs=H_s_z, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=3)
			H_s_z = residual_block_dense(inputs=H_s_z, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=4)
			H_s_z = residual_block_dense(inputs=H_s_z, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=5)
			s_z = dense(inputs=H_s_z, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=5)		

		# J: MLP for encodings, 8 layers or 4 ResNet and 2048 dim according to BigBiGAN.
		with tf.variable_scope('discriminator_J', reuse=reuse):
			if display: print('Discriminator_J:')
			J_s_xz = tf.concat([F_s_x, H_s_z], axis=1, name='concatenate_network_outputs')
			J_s_xz = residual_block_dense(inputs=J_s_xz, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=1)
			J_s_xz = residual_block_dense(inputs=J_s_xz, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=2)
			J_s_xz = residual_block_dense(inputs=J_s_xz, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=3)
			J_s_xz = residual_block_dense(inputs=J_s_xz, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=4)
			J_s_xz = residual_block_dense(inputs=J_s_xz, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=5)
			s_xz = dense(inputs=J_s_xz, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=5)		

	print()
	# output = None
	# s_x = None
	# s_z = None
	return output, s_x, s_z, s_xz