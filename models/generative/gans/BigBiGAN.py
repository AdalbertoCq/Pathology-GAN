import tensorflow as tf
from data_manipulation.utils import *
from models.evaluation.features import *
from models.generative.ops import *
from models.generative.utils import *
from models.generative.tools import *
from models.generative.loss import *
from models.generative.regularizers import *
from models.generative.activations import *
from models.generative.normalization import *
from models.generative.evaluation import *
from models.generative.optimizer import *
from models.generative.discriminator_bigbigan import *
from models.generative.generator import *
from models.generative.encoder import *
from models.generative.gans.GAN import GAN


class BigBiGAN(GAN):
	def __init__(self,
				data,                        # Dataset class, training and test data.
				z_dim,	                     # Latent space dimensions.
				use_bn,                      # Batch Normalization flag to control usage in discriminator.
				alpha,                       # Alpha value for LeakyReLU.
				beta_1,                      # Beta 1 value for Adam Optimizer.
				learning_rate_g,             # Learning rate generator.
				learning_rate_d,             # Learning rate discriminator.
				learning_rate_e,             # Learning rate encoder.
				layers=5,					 # Number for layers for Generator/Discriminator.
				power_iterations=1,          # Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
				beta_2=None,                 # Beta 2 value for Adam Optimizer.
				n_critic=1,                  # Number of batch gradient iterations in Discriminator per Generator.
				gp_coeff=.5,                 # Gradient Penalty coefficient for the Wasserstein Gradient Penalty loss.
				conditional=False,			 # Conditional GAN flag.
				label_dim=None,              # Label space dimensions.
				num_classes=2,				 # Label number of different classes.
				label_t='cat',				 # Type of label: Categorical, Continuous.
				init = 'orthogonal',		 	 # Weight Initialization: Orthogonal in BigGAN.
				loss_type='hinge',  	     # Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.
				regularizer_scale=1e-4,         # Orthogonal regularization.
				model_name='BigGAN'          # Name to give to the model.
				):

		self.layers = layers
		# Training parameters
		self.power_iterations = power_iterations
		self.gp_coeff = gp_coeff
		self.beta_2 = beta_2
		self.regularizer_scale = regularizer_scale
		self.n_sing = 4
		self.label_dim = label_dim
		self.num_classes = num_classes
		if conditional:
			self.normalization = conditional_batch_norm
		else:
			self.normalization = conditional_instance_norm
			self.normalization = conditional_batch_norm
		self.learning_rate_e = learning_rate_e

		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, 
						 conditional=conditional, num_classes=num_classes, label_t=label_t, n_critic=n_critic, init=init, loss_type=loss_type, model_name=model_name)

	def discriminator(self, images, encoding, reuse, init, label_input=None):
		output, s_x, s_z, s_xz = discriminator_resnet(images=images, encoding=encoding, layers=self.layers, spectral=True, activation=leakyReLU, reuse=reuse, attention=28, init=init, regularizer=orthogonal_reg(self.regularizer_scale), 
											  label=label_input, label_t=self.label_t)
		return output, s_x, s_z, s_xz


	def mapping(self, z_input, z_dim, reuse, is_train, init):
		z_map = mapping_resnet(z_input, z_dim=z_dim, layers=4, reuse=reuse, is_train=is_train, spectral=True, activation=leakyReLU, normalization=batch_norm, init=init, regularizer=orthogonal_reg(self.regularizer_scale))
		return z_map

	def generator(self, z_input, reuse, is_train, init, label_input=None):
		output = generator_resnet(z_input=z_input, image_channels=self.image_channels, layers=self.layers, spectral=True, activation=leakyReLU, reuse=reuse, is_train=is_train, noise_input_f=True, 
								  bigGAN=False, normalization=self.normalization, init=init, regularizer=orthogonal_reg(self.regularizer_scale), cond_label=label_input, attention=28)
		return output


	def encoder(self, images, reuse, is_train, init):
		z_hat = encoder_resnet(images=images, layers=self.layers, z_dim=self.z_dim, spectral=True, is_train=is_train, activation=leakyReLU, reuse=reuse, init=init, regularizer=orthogonal_reg(self.regularizer_scale), 
			normalization=batch_norm, attention=28)
		return z_hat


	def loss(self):
		score_real = [self.s_x_real, self.s_z_real, self.s_xz_real]
		score_fake = [self.s_x_fake, self.s_z_fake, self.s_xz_fake]
		loss_dis, loss_gen, loss_dis_s_x, loss_gen_s_x, loss_dis_s_z, loss_gen_s_z, loss_dis_s_xz, loss_gen_s_xz = loss_RaBigBiGAN(score_real=score_real, score_fake=score_fake, real_images=self.real_images, fake_images=self.fake_images, real_encoding=self.z_input, fake_encoding=self.z_hat, 
									discriminator=self.discriminator, gp_coeff=self.gp_coeff, init=self.init)
		return loss_dis, loss_gen, loss_dis_s_x, loss_gen_s_x, loss_dis_s_z, loss_gen_s_z, loss_dis_s_xz, loss_gen_s_xz

	def optimization(self):
		train_discriminator, train_generator, train_encoder = optimizer_RaBigBiGAN(loss_gen=self.loss_gen, loss_dis=self.loss_dis, learning_rate_input_g=self.learning_rate_input_g, 
																learning_rate_input_d=self.learning_rate_input_d, learning_rate_input_e=self.learning_rate_input_e, beta_1=self.beta_1, beta_2=self.beta_2)
		return train_discriminator, train_generator, train_encoder

	def model_inputs(self):
		real_images = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images')
		z_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input')
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')
		learning_rate_e = tf.placeholder(dtype=tf.float32, name='learning_rate_e')
		if self.conditional:
			label_input = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes), name='label_input')
		else:
			label_input = None
		return real_images, z_input, learning_rate_g, learning_rate_d, learning_rate_e, label_input

	def build_model(self):

		with tf.device('/gpu:1'):
			# Inputs.
			self.real_images, self.z_input, self.learning_rate_input_g, self.learning_rate_input_d, self.learning_rate_input_e, self.label_input = self.model_inputs()

			# Mapping network
			self.z_map = self.mapping(self.z_input, z_dim=self.z_dim, reuse=False, is_train=True, init=self.init)
			self.z_map_out = self.mapping(self.z_input, z_dim=self.z_dim, reuse=True, is_train=False, init=self.init)

		with tf.device('/gpu:0'):
			# Generator:
			self.fake_images = self.generator(self.z_map, reuse=False, is_train=True, init=self.init, label_input=self.label_input)
			self.output_gen = self.generator(self.z_map_out, reuse=True, is_train=False, init=self.init, label_input=self.label_input)

		with tf.device('/gpu:1'):
			# Encoder:
			# self.mus, self.stds = self.encoder(images=self.real_images, reuse=False, is_train=True, init=self.init)
			# self.z_hat = self.mus + (self.stds*tf.random.normal(shape=()))
			self.z_hat = self.encoder(images=self.real_images, reuse=False, is_train=True, init=self.init)
			self.z_hat_out = self.encoder(images=self.real_images, reuse=True, is_train=False, init=self.init)

		with tf.device('/gpu:2'):
			# Discriminator
			self.output_fake, self.s_x_fake, self.s_z_fake, self.s_xz_fake = self.discriminator(images=self.fake_images, encoding=self.z_map, reuse=False, init=self.init, label_input=self.label_input) 
			self.output_real, self.s_x_real, self.s_z_real, self.s_xz_real = self.discriminator(images=self.real_images, encoding=self.z_hat, reuse=True, init=self.init, label_input=self.label_input)

		with tf.device('/gpu:2'):
			# Loss and Optimizer
			self.loss_dis, self.loss_gen, self.loss_dis_s_x, self.loss_gen_s_x, self.loss_dis_s_z, self.loss_gen_s_z, self.loss_dis_s_xz, self.loss_gen_s_xz = self.loss()
		
		with tf.device('/gpu:3'):	
			self.train_discriminator, self.train_generator, self.train_encoder = self.optimization()


	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False, tracking=False, evaluation=None):
		run_epochs = 0    
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints, csvs = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)
		losses = ['Discriminator Loss', 'Generator Loss', 'Dis s_x', 'Gen s_x', 'Dis s_z', 'Gen s_z', 'Dis s_xz', 'Gen s_xz']
		# losses = ['Discriminator Loss', 'Generator Loss']
		setup_csvs(csvs=csvs, model=self, losses=losses)
		report_parameters(self, epochs, restore, data_out_path)
		
		steady_latent = np.random.normal(size=(self.batch_size, self.z_dim)) 

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as session:
			session.run(tf.global_variables_initializer())
			if restore:
				check = get_checkpoint(data_out_path)
				saver.restore(session, check)
				print('Restored model: %s' % check)

			writer = tf.summary.FileWriter(os.path.join(data_out_path, 'tensorboard'), graph_def=session.graph_def)	
			for epoch in range(1, epochs+1):
				saver.save(sess=session, save_path=checkpoints)
				for batch_images, batch_labels in data.training:
					# Inputs.
					z_batch = np.random.normal(size=(self.batch_size, self.z_dim)) 
					feed_dict = {self.z_input:z_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d, self.learning_rate_input_e: self.learning_rate_e}
					if self.conditional:
						batch_labels = np.reshape(batch_labels[:, self.label_dim], (-1, 1))
						if self.label_dim==0:
							batch_labels = survival_5(batch_labels)
						batch_labels = tf.keras.utils.to_categorical(y=batch_labels, num_classes=2)
						feed_dict[self.label_input] = batch_labels

					# Update critic.
					session.run([self.train_discriminator], feed_dict=feed_dict)
					
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run([self.train_encoder], feed_dict=feed_dict)
						session.run([self.train_generator], feed_dict=feed_dict)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						feed_dict = {self.z_input:z_batch, self.real_images:batch_images}
						if self.conditional:
							feed_dict[self.label_input] = batch_labels
						epoch_loss_dis, epoch_loss_gen, epoch_loss_dis_s_x, epoch_loss_gen_s_x, epoch_loss_dis_s_z, epoch_loss_gen_s_z, epoch_loss_dis_s_xz, epoch_loss_gen_s_xz = session.run([self.loss_dis, self.loss_gen, self.loss_dis_s_x, self.loss_gen_s_x, self.loss_dis_s_z, self.loss_gen_s_z, self.loss_dis_s_xz, self.loss_gen_s_xz], feed_dict=feed_dict)
						update_csv(model=self, file=csvs[0], variables=[epoch_loss_dis, epoch_loss_gen, epoch_loss_dis_s_x, epoch_loss_gen_s_x, epoch_loss_dis_s_z, epoch_loss_gen_s_z, epoch_loss_dis_s_xz, epoch_loss_gen_s_xz], epoch=epoch, iteration=run_epochs, losses=losses)
						# epoch_loss_dis, epoch_loss_gen = session.run([self.loss_dis, self.loss_gen], feed_dict=feed_dict)
						# update_csv(model=self, file=csvs[0], variables=[epoch_loss_dis, epoch_loss_gen], epoch=epoch, iteration=run_epochs, losses=losses)
						if tracking:
							f_sing_gen, f_sing_dis = filter_singular_values(model=self, n_sing=self.n_sing)
							jac_sign_values = jacobian_singular_values(session=session, model=self, z_batch=z_batch)
							update_csv(model=self, file=csvs[1], variables=[f_sing_gen, f_sing_dis], epoch=epoch, iteration=run_epochs)
							update_csv(model=self, file=csvs[2], variables=jac_sign_values, epoch=epoch, iteration=run_epochs)

					if show_epochs is not None and run_epochs % show_epochs == 0:
						gen_samples, _ = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, label_input=self.label_input, labels=batch_labels, output_fake=self.output_gen, n_images=25, show=False)
						write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_iter_%s.png' % run_epochs), data=gen_samples, metadata=False)

					run_epochs += 1
				data.training.reset()

				# After each epoch dump a sample of generated images.
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)
				feed_dict = {self.z_input:steady_latent, self.real_images:batch_images}
				gen_samples = session.run([self.output_gen], feed_dict=feed_dict)[0]
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_steady_epoch_%s.png' % epoch), data=gen_samples, metadata=False)

				# if evaluation is not None and epoch >= evaluation:
					# generate_samples_epoch(session=session, model=self, data_shape=data.test.shape[1:], epoch=epoch, evaluation_path=os.path.join(data_out_path, 'evaluation'))
