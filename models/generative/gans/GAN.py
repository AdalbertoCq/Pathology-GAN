import tensorflow as tf
from models.generative.ops import *
from models.generative.utils import *
from models.generative.loss import *
from models.generative.evaluation import *
from models.generative.optimizer import *


class GAN:
	def __init__(self, 
				data,                        # Dataset class, training and test data.
				z_dim,                       # Latent space dimensions.
				use_bn,                      # Batch Normalization flag to control usage in discriminator.
				alpha,                       # Alpha value for LeakyReLU.
				beta_1,                      # Beta 1 value for Adam Optimizer.
				learning_rate_g,             # Learning rate generator.
				learning_rate_d,             # Learning rate discriminator.
				conditional=False,			 # Conditional GAN flag.
				num_classes=None,              # Label space dimensions.
				label_t='cat',				 # Type of label: Categorical, Continuous.
				n_critic=1,                  # Number of batch gradient iterations in Discriminator per Generator.
				init = 'xavier',			 # Weight Initialization: Orthogonal in BigGAN.
				loss_type = 'standard',      # Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.				
				model_name='GAN'          	 # Name to give to the model.
				):

		# Loss function definition.
		self.loss_type = loss_type
		self.model_name = model_name

		# Input data variables.
		self.image_height = data.training.patch_h
		self.image_width = data.training.patch_w
		self.image_channels = data.training.n_channels
		self.batch_size = data.training.batch_size

		# Latent space dimensions.
		self.z_dim = z_dim
		self.num_classes = num_classes
		self.label_t = label_t
		self.conditional = conditional

		# Network details
		self.use_bn = use_bn
		self.alpha = alpha
		self.init = init

		# Training parameters
		self.n_critic = n_critic
		self.beta_1 = beta_1
		self.learning_rate_g = learning_rate_g
		self.learning_rate_d = learning_rate_d

		self.build_model()

		self.gen_filters, self.dis_filters = gather_filters()


	def discriminator(self, images, reuse, init, label_input=None):
		with tf.variable_scope('discriminator', reuse=reuse):
			# Padding = 'Same' -> H_new = H_old // Stride

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

		return output, logits

	def generator(self, z_input, reuse, is_train, init, label_input=None):
		with tf.variable_scope('generator', reuse=reuse):
			# Doesn't work ReLU, tried.
			# Input Shape = (None, 100)
			# Dense.
			net = dense(inputs=z_input, out_dim=1024, scope=1)				
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

			logits = convolutional(inputs=net, output_channels=self.image_channels, filter_size=4, stride=2, padding='SAME', conv_type='transpose', scope=4)
			# Shape = (None, 56, 56, 3)
			output = tf.nn.sigmoid(x=logits, name='output')
		return output


	def model_inputs(self):
		real_images = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images')
		z_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input')
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')
		if self.conditional:
			label_input = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes), name='label_input')
		else:
			label_input = None
		return real_images, z_input, learning_rate_g, learning_rate_d, label_input


	def loss(self):
		loss_dis, loss_gen = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real)
		return loss_dis, loss_gen


	def optimization(self):
		train_discriminator, train_generator = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_g, self.learning_rate_input_d)
		return train_discriminator, train_generator
    

	def build_model(self):
		# Inputs.
		self.real_images, self.z_input, self.learning_rate_input_g, self.learning_rate_input_d, self.label_input = self.model_inputs()

		# Neural Network Generator and Discriminator.
		self.fake_images = self.generator(self.z_input, reuse=False, is_train=True, init=self.init, label_input=self.label_input)
		self.output_fake, self.logits_fake = self.discriminator(images=self.fake_images, reuse=False, init=self.init, label_input=self.label_input) 
		self.output_real, self.logits_real = self.discriminator(images=self.real_images, reuse=True, init=self.init, label_input=self.label_input)

		# Losses.
		self.loss_dis, self.loss_gen = self.loss()

		# Optimizers.
		self.train_discriminator, self.train_generator = self.optimization()

		# Generator for output sampling.
		self.output_gen = self.generator(self.z_input, reuse=True, is_train=False, init=self.init, label_input=self.label_input)


	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False, tracking=False):
		run_epochs = 0    
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)
		losses = ['Generator Loss', 'Discriminator Loss']
		setup_csvs(csvs=csvs, model=self, losses=losses)
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
					feed_dict = {self.z_input:z_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}
					if self.conditional:
						feed_dict[self.label_input] = labels_to_binary(labels, n_bits=self.num_classes)

					# Update critic.
					session.run(self.train_discriminator, feed_dict=feed_dict)	
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run(self.train_generator, feed_dict=feed_dict)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						feed_dict = {self.z_input:z_batch, self.real_images:batch_images}
						if self.conditional:
							feed_dict[self.label_input] = labels
						epoch_loss_dis, epoch_loss_gen = session.run([self.loss_dis, self.loss_gen], feed_dict=feed_dict)
						update_csv(model=self, file=csvs[0], variables=[epoch_loss_gen, epoch_loss_dis], epoch=epoch, iteration=run_epochs, losses=losses)
						if tracking:
							f_sing_gen, f_sing_dis = filter_singular_values(self)
							jac_sign_values = jacobian_singular_values(session=session, model=self, z_batch=z_batch)
							update_csv(model=self, file=csvs[1], variables=[f_sing_gen, f_sing_dis], epoch=epoch, iteration=run_epochs)
							update_csv(model=self, file=csvs[2], variables=jac_sign_values, epoch=epoch, iteration=run_epochs)
						
					if show_epochs is not None and run_epochs % show_epochs == 0:
						gen_samples, sample_z = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=n_images)
						if save_img:
							img_storage[run_epochs//show_epochs] = gen_samples
							latent_storage[run_epochs//show_epochs] = sample_z
					
					run_epochs += 1
				data.training.reset()

				gen_samples, _ = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=25, show=False)
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)


