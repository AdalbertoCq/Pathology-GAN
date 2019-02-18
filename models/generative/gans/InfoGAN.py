import tensorflow as tf
from models.generative.ops import *
from models.generative.utils import *
from models.generative.gans.GAN import GAN
from models.generative.loss import losses

class InfoGAN(GAN):
	def __init__(self, 
				data,                        		# Dataset class, training and test data.
				z_dim,                       		# Latent space dimensions.
				c_dim,                       		# C Latent space dimensions.
				use_bn,                     		# Batch Normalization flag to control usage in discriminator.
				alpha,                       		# Alpha value for LeakyReLU.
				beta_1,                     		# Beta 1 value for Adam Optimizer.
				learning_rate_g,             		# Learning rate generator.
				learning_rate_d,             		# Learning rate discriminator.
				delta, 						 		# Delta hyperparameter.
				loss_type = 'standard infogan',     # Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.				
				model_name = 'InfoGAN'         		# Name to give to the model.
				):

		self.c_dim = c_dim
		self.delta = delta
		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d,
						 loss_type=loss_type, model_name=model_name)


	def discriminator(self, images, reuse):
		with tf.variable_scope('discriminator', reuse=reuse):
			# Padding = 'Same' -> H_new = H_old // Stride
			# Input Shape = (None, 56, 56, 3)

			# Conv.
			net = convolutional(inputs=images, output_channels=64, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=1)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 28, 28, 64)

			# Conv.
			net = convolutional(inputs=net, output_channels=128, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=2)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 14, 14, 128)

			# Conv.
			net = convolutional(inputs=net, output_channels=256, filter_size=5, stride=2, padding='SAME', conv_type='convolutional', scope=3)
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 7, 7, 256)

			# Flatten.
			net = tf.layers.flatten(inputs=net)
			# Shape = (None, 7*7*256)

			# Dense.
			net = dense(inputs=net, out_dim=1024, scope=1)				
			if self.use_bn: net = tf.layers.batch_normalization(inputs=net, training=True)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 1024)

			# Dense
			logits = dense(inputs=net, out_dim=1, scope=2)				
			# Shape = (None, 1)
			output = tf.nn.sigmoid(x=logits)

			mean_c_x = dense(inputs=net, out_dim=self.c_dim, scope=3)
			logs2_c_x = dense(inputs=net, out_dim=self.c_dim, scope=4)

		return output, logits, mean_c_x, logs2_c_x


	def generator(self, z_input, c_input, reuse, is_train):
		with tf.variable_scope('generator', reuse=reuse):
			# Doesn't work ReLU, tried.
			# Z Input Shape = (None, 100)
			# C Input Shape = (None, 20)
			net = tf.concat([z_input, c_input], axis=1)

			# Dense.
			net = dense(inputs=net, out_dim=1024, scope=1)				
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 1024)

			# Dense.
			net = dense(inputs=net, out_dim=256*7*7, scope=2)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 256*7*7)

			# Reshape
			net = tf.reshape(tensor=net, shape=(-1, 7, 7, 256), name='reshape')
			# Shape = (None, 7, 7, 256)

			# Conv.
			net = convolutional(inputs=net, output_channels=256, filter_size=4, stride=2, padding='SAME', conv_type='transpose', scope=1)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 14, 14, 256)

			# Conv.
			net = convolutional(inputs=net, output_channels=128, filter_size=5, stride=1, padding='SAME', conv_type='convolutional', scope=2)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 14, 14, 128)

			# Conv.
			net = convolutional(inputs=net, output_channels=128, filter_size=4, stride=2, padding='SAME', conv_type='transpose', scope=3)
			net = tf.layers.batch_normalization(inputs=net, training=is_train)
			net = leakyReLU(net, self.alpha)
			# Shape = (None, 28, 28, 128)

			logits = convolutional(inputs=net, output_channels=self.image_channels, filter_size=4, stride=2, padding='SAME', conv_type='transpose', scope=4)
			# Shape = (None, 56, 56, 3)
			output = tf.nn.sigmoid(x=logits, name='output')
		return output
	
	def model_inputs(self):
		real_images = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images')
		z_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input')
		c_input = tf.placeholder(dtype=tf.float32, shape=(None, self.c_dim), name='c_input')
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')
		return real_images, z_input, c_input, learning_rate_g, learning_rate_d


	def loss(self):
		loss_dis, loss_gen, mututal_loss = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, mean_c_x_fake=self.mean_c_x_fake, 
									logs2_c_x_fake=self.logs2_c_x_fake, input_c=self.c_input, delta=self.delta)
		return loss_dis, loss_gen, mututal_loss


	def build_model(self):
		# Inputs.
		self.real_images, self.z_input, self.c_input, self.learning_rate_input_g, self.learning_rate_input_d = self.model_inputs()

		# Neural Network Generator and Discriminator.
		self.fake_images = self.generator(self.z_input, self.c_input, reuse=False, is_train=True)
		self.output_fake, self.logits_fake, self.mean_c_x_fake, self.logs2_c_x_fake = self.discriminator(images=self.fake_images, reuse=False) 
		self.output_real, self.logits_real, _, _ = self.discriminator(images=self.real_images, reuse=True)

		# Losses.
		self.loss_dis, self.loss_gen, self.mututal_loss = self.loss()

		# Optimizers.
		self.train_discriminator, self.train_generator = self.optimization()

		self.output_gen = self.generator(self.z_input, self.c_input, reuse=True, is_train=False)


	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False):
		run_epochs = 0    
		losses = list()
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)
		report_parameters(self, epochs, restore, data_out_path)

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			if restore:
				check = get_checkpoint(data_out_path)
				saver.restore(session, check)
				print('Restored model: %s' % check)
			for epoch in range(1, epochs+1):
				saver.save(sess=session, save_path=checkpoints)
				for batch_images, batch_labels in data.training:
					# Inputs.
					z_batch = np.random.uniform(low=-1., high=1., size=(self.batch_size, self.z_dim))
					c_batch = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.c_dim))             
					feed_dict = {self.z_input:z_batch, self.c_input:c_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, 
								 self.learning_rate_input_d: self.learning_rate_d}

					# Update critic.
					session.run(self.train_discriminator, feed_dict=feed_dict)	
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run(self.train_generator, feed_dict=feed_dict)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						feed_dict = {self.z_input:z_batch, self.c_input:c_batch, self.real_images:batch_images}
						epoch_loss_dis, epoch_loss_gen, epoch_loss_mut = session.run([self.loss_dis, self.loss_gen, self.mututal_loss], feed_dict=feed_dict)
						losses.append((epoch_loss_dis, epoch_loss_gen))
						print('Epochs %s/%s: Generator Loss: %s. Discriminator Loss: %s Mutual Information Loss: %s' % 
							(epoch, epochs, np.round(epoch_loss_gen, 4), np.round(epoch_loss_dis, 4), np.round(epoch_loss_mut, 4)))
					if show_epochs is not None and run_epochs % show_epochs == 0:
						gen_samples, sample_z = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=n_images, 
															   c_input=self.c_input, c_dim=self.c_dim)
						if save_img:
							img_storage[run_epochs//show_epochs] = gen_samples
							latent_storage[run_epochs//show_epochs] = sample_z

					run_epochs += 1
				data.training.reset()
		save_loss(losses, data_out_path, dim=30)
