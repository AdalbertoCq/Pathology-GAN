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


class SAGAN(GAN):
	def __init__(self,
				data,                        # Dataset class, training and test data.
				z_dim,                       # Latent space dimensions.
				use_bn,                      # Batch Normalization flag to control usage in discriminator.
				alpha,                       # Alpha value for LeakyReLU.
				beta_1,                      # Beta 1 value for Adam Optimizer.
				learning_rate_g,             # Learning rate generator.
				learning_rate_d,             # Learning rate discriminator.
				power_iterations=1,          # Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
				beta_2=None,                 # Beta 2 value for Adam Optimizer.
				n_critic=1,                  # Number of batch gradient iterations in Discriminator per Generator.
				gp_coeff=.5,                 # Gradient Penalty coefficient for the Wasserstein Gradient Penalty loss.
				conditional=False,			 # Conditional GAN flag.
				label_dim=None,              # Label space dimensions.
				label_t='cat',				 # Type of label: Categorical, Continuous.
				init = 'xavier',			 # Weight Initialization: Orthogonal in BigGAN.
				loss_type='standard',        # Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.
				model_name='SAGAN'           # Name to give to the model.
				):

		# Training parameters
		self.power_iterations = power_iterations
		self.gp_coeff = gp_coeff
		self.beta_2 = beta_2
		self.n_sing = 4
		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, 
						 conditional=conditional, label_dim=label_dim, label_t=label_t, n_critic=n_critic, init=init, loss_type=loss_type, model_name=model_name)

	def discriminator(self, images, reuse, init, label_input=None):
		output, logits = discriminator_resnet(images=images, layers=5, spectral=True, activation=leakyReLU, reuse=reuse, attention=28, init=init, label=label_input, label_t=self.label_t)
		return output, logits

	def generator(self, z_input, reuse, is_train, init, label_input=None):
		output = generator_resnet(z_input=z_input, image_channels=self.image_channels, layers=5, spectral=True, activation=leakyReLU, reuse=reuse, is_train=is_train, 
								  normalization=conditional_instance_norm, init=init, cond_label=label_input, attention=28)
		return output

	def loss(self):
		loss_dis, loss_gen = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, real_images=self.real_images, fake_images=self.fake_images, 
									discriminator=self.discriminator, gp_coeff=self.gp_coeff)
		return loss_dis, loss_gen

	def optimization(self):
		train_discriminator, train_generator = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_g, self.learning_rate_input_d, beta_2=self.beta_2)
		return train_discriminator, train_generator

	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False, tracking=False):
		run_epochs = 0    
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints, csvs = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)
		losses = ['Generator Loss', 'Discriminator Loss']
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
					feed_dict = {self.z_input:z_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}
					if self.conditional:
						batch_labels = labels_to_binary(batch_labels, n_bits=self.label_dim)
						feed_dict[self.label_input] = batch_labels

					# Update critic.
					session.run([self.train_discriminator], feed_dict=feed_dict)
					
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run([self.train_generator], feed_dict=feed_dict)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						feed_dict = {self.z_input:z_batch, self.real_images:batch_images}
						if self.conditional:
							feed_dict[self.label_input] = batch_labels
						epoch_loss_dis, epoch_loss_gen = session.run([self.loss_dis, self.loss_gen], feed_dict=feed_dict)
						update_csv(model=self, file=csvs[0], variables=[epoch_loss_gen, epoch_loss_dis], epoch=epoch, iteration=run_epochs, losses=losses)
						if tracking:
							f_sing_gen, f_sing_dis = filter_singular_values(model=self, n_sing=self.n_sing)
							jac_sign_values = jacobian_singular_values(session=session, model=self, z_batch=z_batch)
							update_csv(model=self, file=csvs[1], variables=[f_sing_gen, f_sing_dis], epoch=epoch, iteration=run_epochs)
							update_csv(model=self, file=csvs[2], variables=jac_sign_values, epoch=epoch, iteration=run_epochs)

					if show_epochs is not None and run_epochs % show_epochs == 0:
						gen_samples, sample_z = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, label_input=self.label_input, labels=batch_labels, output_fake=self.output_gen, n_images=n_images, dim=30)
						if save_img:
							img_storage[run_epochs//show_epochs] = gen_samples
							latent_storage[run_epochs//show_epochs] = sample_z
					
					run_epochs += 1
				data.training.reset()

				gen_samples, _ = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, label_input=self.label_input, labels=batch_labels, output_fake=self.output_gen, n_images=25, show=False)
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)
