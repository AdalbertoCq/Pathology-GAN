from data_manipulation.utils import write_sprite_image
from tensorflow.contrib.tensorboard.plugins import projector
from models.generative.utils import *


# Method to generate random samples from a model, it also dumps a sprite image width them.
def generate_samples(model, n_images, data_out_path):
		saver = tf.train.Saver()
		with tf.Session() as session:
			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			check = get_checkpoint(data_out_path)
			saver.restore(session, check)
			# Sample images.
			gen_samples, sample_z = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, n_images=n_images, show=False)

		# Dump images into sprite.
		write_sprite_image(os.path.join(data_out_path, 'geneated_samples.png'), gen_samples, metadata=False)

		return gen_samples, sample_z


# Method to generate images from the linear interpolation of two latent space vectors.
def linear_interpolation(model, n_images, data_out_path, orig_vector, dest_vector):
	saver = tf.train.Saver()
	with tf.Session() as session:
		# Initializer and restoring model.
		session.run(tf.global_variables_initializer())
		check = get_checkpoint(data_out_path)
		saver.restore(session, check)

		sequence = np.zeros((n_images, model.z_dim))
		# Generate images from model. 
		alphaValues = np.linspace(0, 1, n_images)
		for i, alpha in enumerate(alphaValues):
			sequence[i, :] = orig_vector*(1-alpha) + dest_vector*alpha
			# Latent space interpolation

		print(sequence.shape)
		feed_dict = {model.z_input: sequence}
		linear_interpolation = session.run(model.output_gen, feed_dict=feed_dict)
		print(linear_interpolation.shape)

	return linear_interpolation, sequence


# Generates samples from the latent space to show in tensorboard. 
# Restores a model and somples from it.
def run_latent(model, n_images, data_out_path):
	
	tensorboard_path = os.path.join(data_out_path, 'tensorboard')
	saver = tf.train.Saver()
	with tf.Session() as session:

		# Inputs for tensorboard.
 		tf_data = tf.Variable(tf.zeros((n_images, model.z_dim)), name='tf_data')
		input_sample = tf.placeholder(tf.float32, shape=(n_images, model.z_dim))
		set_tf_data = tf.assign(tf_data, input_sample, validate_shape=False)

		# Initializer and restoring model.
		session.run(tf.global_variables_initializer())
		check = get_checkpoint(data_out_path)
		saver.restore(session, check)

		# Sample images.
		gen_samples, sample_z = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, n_images=n_images, show=False)

		# Generate sprite of images.
		write_sprite_image(os.path.join(data_out_path, 'gen_sprite.png'), gen_samples)
		
		# Variable for embedding.
		saver_latent = tf.train.Saver([tf_data])
		session.run(set_tf_data, feed_dict={input_sample: sample_z})
		saver_latent.save(sess=session, save_path=os.path.join(tensorboard_path, 'tf_data.ckpt'))

		# Tensorflow embedding.
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = tf_data.name
		embedding.metadata_path = os.path.join(data_out_path, 'metadata.tsv')
		embedding.sprite.image_path = os.path.join(data_out_path, 'gen_sprite.png')
		embedding.sprite.single_image_dim.extend([self.image_height, self.image_width])
		projector.visualize_embeddings(tf.summary.FileWriter(tensorboard_path), config)	