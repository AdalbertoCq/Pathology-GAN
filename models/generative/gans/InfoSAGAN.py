import tensorflow as tf
import tensorflow_probability as tfp
from data_manipulation.utils import *
from models.generative.ops import *
from models.generative.utils import *
from models.generative.tools import *
from models.generative.loss import *
from models.generative.activations import *
from models.generative.normalization import *
from models.generative.evaluation import *
from models.generative.optimizer import *
from models.generative.discriminator import *
from models.generative.generator import *
from models.generative.gans.GAN import GAN


class InfoSAGAN(GAN):
	def __init__(self,
				data,                        	# Dataset class, training and test data.
				z_dim,                       	# Latent space dimensions.
				c_dim,                       	# C Latent space dimensions.
				use_bn,                     	# Batch Normalization flag to control usage in discriminator.
				alpha,                       	# Alpha value for LeakyReLU.
				beta_1,                      	# Beta 1 value for Adam Optimizer.
				learning_rate_g,             	# Learning rate generator.
				learning_rate_d,             	# Learning rate discriminator.
				delta=1, 					 	# Delta hyperparameter.
				power_iterations=1,          	# Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
				beta_2=None,                 	# Beta 2 value for Adam Optimizer.
				n_critic=1,                  	# Number of batch gradient iterations in Discriminator per Generator.
				gp_coeff=.5,                 	# Gradient Penalty coefficient for the Wasserstein Gradient Penalty loss.
				loss_type='standard infogan',   # Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.
				model_name='InfoSAGAN'          
				# Name to give to the model.
				):


		# InfoGAN param.
		self.c_dim = c_dim
		self.delta = delta

		# Training parameters
		self.power_iterations = power_iterations
		self.gp_coeff = gp_coeff
		self.beta_2 = beta_2

		# Singular values #
		self.n_sing = 4
		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, n_critic=n_critic, loss_type=loss_type, model_name=model_name)
		
	def discriminator(self, images, reuse):
		output, logits, mean_c_x, logs2_c_x = discriminator_resnet(images=images, layers=5, spectral=True, activation=leakyReLU, reuse=reuse, attention=28, infoGAN=True, c_dim=self.c_dim)
		return output, logits, mean_c_x, logs2_c_x

	def generator(self, z_input, c_input, reuse, is_train):
		output = generator_resnet(z_input=z_input, image_channels=self.image_channels, layers=5, spectral=True, activation=leakyReLU, reuse=reuse, is_train=is_train, normalization=conditional_instance_norm, 
								  cond_label=c_input, attention=28)
		return output

	def model_inputs(self):
		real_images = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images')
		z_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='z_input')
		c_input = tf.placeholder(dtype=tf.float32, shape=(None, self.c_dim), name='c_input')
		learning_rate_g = tf.placeholder(dtype=tf.float32, name='learning_rate_g')
		learning_rate_d = tf.placeholder(dtype=tf.float32, name='learning_rate_d')
		return real_images, z_input, c_input, learning_rate_g, learning_rate_d

	def loss(self):
		loss_dis, loss_gen, mututal_loss = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, real_images=self.real_images, 
												  fake_images=self.fake_images, discriminator=self.discriminator, gp_coeff=self.gp_coeff, mean_c_x_fake=self.mean_c_x_fake, 
												  logs2_c_x_fake=self.logs2_c_x_fake, input_c=self.c_input, delta=self.delta)
		return loss_dis, loss_gen, mututal_loss

	def optimization(self):
		train_discriminator, train_generator = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_g, self.learning_rate_input_d, beta_2=self.beta_2)
		return train_discriminator, train_generator

	def build_model(self):
		# Inputs.
		self.real_images, self.z_input, self.c_input, self.learning_rate_input_g, self.learning_rate_input_d = self.model_inputs()

		# Neural Network Generator and Discriminator.
		self.fake_images = self.generator(z_input=self.z_input, c_input=self.c_input, reuse=False, is_train=True)
		self.output_fake, self.logits_fake, self.mean_c_x_fake, self.logs2_c_x_fake = self.discriminator(images=self.fake_images, reuse=False) 
		self.output_real, self.logits_real, _, _ = self.discriminator(images=self.real_images, reuse=True)

		# Losses.
		self.loss_dis, self.loss_gen, self.mututal_loss = self.loss()

		# Optimizers.
		self.train_discriminator, self.train_generator = self.optimization()

		self.output_gen = self.generator(self.z_input, self.c_input, reuse=True, is_train=False)

	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False, tracking=False):
		run_epochs = 0    
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints, csvs = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)
		losses = ['Generator Loss', 'Discriminator Loss', 'Mutual Information Loss']
		setup_csvs(csvs=csvs, model=self, losses=losses)
		report_parameters(self, epochs, restore, data_out_path)

		with tf.Session() as session:
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
					z_batch = np.random.uniform(low=-1., high=1., size=(self.batch_size, self.z_dim))               
					c_batch = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.c_dim))             
					feed_dict = {self.z_input:z_batch, self.c_input:c_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}
					# Update critic.
					session.run([self.train_discriminator], feed_dict=feed_dict)
					
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run([self.train_generator], feed_dict=feed_dict)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						feed_dict = {self.z_input:z_batch, self.c_input:c_batch, self.real_images:batch_images}
						epoch_loss_dis, epoch_loss_gen, epoch_loss_mut = session.run([self.loss_dis, self.loss_gen, self.mututal_loss], feed_dict=feed_dict)
						update_csv(model=self, file=csvs[0], variables=[epoch_loss_gen, epoch_loss_dis, epoch_loss_mut], epoch=epoch, iteration=run_epochs, losses=losses)
						if tracking:
							f_sing_gen, f_sing_dis = filter_singular_values(model=self, n_sing=self.n_sing)
							jac_sign_values = jacobian_singular_values(session=session, model=self, z_batch=z_batch)
							update_csv(model=self, file=csvs[1], variables=[f_sing_gen, f_sing_dis], epoch=epoch, iteration=run_epochs)
							update_csv(model=self, file=csvs[2], variables=jac_sign_values, epoch=epoch, iteration=run_epochs)

					if show_epochs is not None and run_epochs % show_epochs == 0:
						gen_samples, sample_z = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=n_images, c_input=self.c_input, c_dim=self.c_dim, dim=30)
						if save_img:
							img_storage[run_epochs//show_epochs] = gen_samples
							latent_storage[run_epochs//show_epochs] = sample_z
					run_epochs += 1
				data.training.reset()

				gen_samples, _ = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, c_input=self.c_input, c_dim=self.c_dim, n_images=25, show=False)
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)
