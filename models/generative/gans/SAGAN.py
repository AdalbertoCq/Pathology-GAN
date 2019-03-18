import tensorflow as tf
import tensorflow_probability as tfp
from models.generative.ops import *
from models.generative.utils import *
from models.generative.tools import *
from models.generative.loss import *
from models.generative.activations import *
from models.generative.normalization import *
from models.generative.optimizer import *
from models.generative.discriminator import *
from models.generative.generator import *
from models.generative.gans.GAN import GAN




'''

Currently missing from the implementation:
	1. TTUR.
	2. Conditional Batch Normalization.
	3. Projection in discriminator.

	Paper parameter values: Adam Optimizer, B1=0, B2=0.9, Lr_disc = 4e-4, Lr_gen=1e-4.

'''

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
				loss_type='standard',        # Loss function type: Standard, Least Square, Wasserstein, Wasserstein Gradient Penalty.
				model_name='SAGAN'           # Name to give to the model.
				):

		# Training parameters
		self.power_iterations = power_iterations
		self.gp_coeff = gp_coeff
		self.beta_2 = beta_2
		self.spec_ops_name = 'SPECTRAL_NORM_UPDATE_OPS'
		super().__init__(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, n_critic=n_critic, loss_type=loss_type, model_name=model_name)
		# self.sing_jacob = check_jacobian_singular()

	def discriminator(self, images, reuse):
		output, logits = discriminator_resnet(images=images, layers=5, spectral=True, activation=leakyReLU, reuse=reuse, attention=28)
		return output, logits

	def generator(self, z_input, reuse, is_train):
		output = generator_resnet(z_input=z_input, image_channels=self.image_channels, layers=5, spectral=True, activation=leakyReLU, reuse=reuse, is_train=is_train, normalization=batch_norm, attention=28)
		return output

	def loss(self):
		loss_dis, loss_gen = losses(self.loss_type, self.output_fake, self.output_real, self.logits_fake, self.logits_real, real_images=self.real_images, 
										fake_images=self.fake_images, discriminator=self.discriminator, gp_coeff=self.gp_coeff)
		return loss_dis, loss_gen

	def optimization(self):
		train_discriminator, train_generator = optimizer(self.beta_1, self.loss_gen, self.loss_dis, self.loss_type, self.learning_rate_input_g, self.learning_rate_input_d, beta_2=self.beta_2)
		return train_discriminator, train_generator

	def check_jacobian_singular(self):
		gen_jacob = tf.gradients(ys=self.fake_images, xs=self.z_input)
		s, _, _ = tf.svd(gen_jacob)
		return s 


	def train(self, epochs, data_out_path, data, restore, show_epochs=100, print_epochs=10, n_images=10, save_img=False):
		run_epochs = 0    
		losses = list() 
		saver = tf.train.Saver()

		img_storage, latent_storage, checkpoints = setup_output(show_epochs, epochs, data, n_images, self.z_dim, data_out_path, self.model_name, restore, save_img)
		report_parameters(self, epochs, restore, data_out_path)

		# Used if we want to update by ops name.
		# sn_update_ops = tf.get_collection(self.spec_ops_name)

		# Track normalized kernels.
		# summary_op = tf.summary.merge_all()

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

					# Normalize weights using the operation collection instead of the control dependencies.
					# for sn_up in sn_update_ops:
					# 	value = session.run(sn_up)

					# Inputs.
					z_batch = np.random.uniform(low=-1., high=1., size=(self.batch_size, self.z_dim))               
					feed_dict = {self.z_input:z_batch, self.real_images:batch_images, self.learning_rate_input_g: self.learning_rate_g, self.learning_rate_input_d: self.learning_rate_d}

					# Update critic.
					# summary, _ = session.run([summary_op, self.train_discriminator], feed_dict=feed_dict)
					# summary = writer.add_summary(summary, run_epochs)
					session.run([self.train_discriminator], feed_dict=feed_dict)
					
					# Update generator after n_critic updates from discriminator.
					if run_epochs%self.n_critic == 0:
						session.run([self.train_generator], feed_dict=feed_dict)

		            # Print losses and Generate samples.
					if run_epochs % print_epochs == 0:
						feed_dict = {self.z_input:z_batch, self.real_images:batch_images}
						epoch_loss_dis, epoch_loss_gen = session.run([self.loss_dis, self.loss_gen], feed_dict=feed_dict)
						losses.append((epoch_loss_dis, epoch_loss_gen))
						print('Epochs %s/%s: Generator Loss: %10s  Discriminator Loss: %10s' % (epoch, epochs, np.round(epoch_loss_gen, 4), np.round(epoch_loss_dis, 4)))
					if show_epochs is not None and run_epochs % show_epochs == 0:
						gen_samples, sample_z = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=n_images, dim=30)
						if save_img:
							img_storage[run_epochs//show_epochs] = gen_samples
							latent_storage[run_epochs//show_epochs] = sample_z
							
					run_epochs += 1
				data.training.reset()

				gen_samples, _ = show_generated(session=session, z_input=self.z_input, z_dim=self.z_dim, output_fake=self.output_gen, n_images=25, show=False)
				write_sprite_image(filename=os.path.join(data_out_path, 'images/gen_samples_epoch_%s.png' % epoch), data=gen_samples, metadata=False)

		save_loss(losses, data_out_path, dim=30)
