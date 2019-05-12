from tensorflow.contrib.tensorboard.plugins import projector
from models.generative.utils import *


# Method to generate random samples from a model, it also dumps a sprite image width them.
def generate_samples(model, n_images, data_out_path, name='geneated_samples.png'):
	saver = tf.train.Saver()
	with tf.Session() as session:
		# Initializer and restoring model.
		session.run(tf.global_variables_initializer())
		check = get_checkpoint(data_out_path)
		saver.restore(session, check)
		# Sample images.
		gen_samples, sample_z = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, n_images=n_images, show=False)

	images_path = os.path.join(data_out_path, 'images')

	# Dump images into sprite.
	# image_sprite = write_sprite_image(filename=os.path.join(images_path, name), data=gen_samples, metadata=False)

	return gen_samples, sample_z

# Method to generate random samples from a model, it also dumps a sprite image width them.
def generate_from_latent(model, latent_vector, data_out_path):
	saver = tf.train.Saver()
	with tf.Session() as session:
		# Initializer and restoring model.
		session.run(tf.global_variables_initializer())
		check = get_checkpoint(data_out_path)
		saver.restore(session, check)
		# Sample images.
		feed_dict = {model.z_input: latent_vector.reshape((-1, model.z_dim))}
		gen_batch = session.run(model.output_gen, feed_dict=feed_dict)
	return gen_batch


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

		feed_dict = {model.z_input: sequence}
		linear_interpolation = session.run(model.output_gen, feed_dict=feed_dict)

	return linear_interpolation, sequence


# Generates samples from the latent space to show in tensorboard. 
# Restores a model and somples from it.
def run_latent(model, n_images, data_out_path, sprite=True):
	
	tensorboard_path = os.path.join(data_out_path, 'tensorboard')
	saver = tf.train.Saver()
	with tf.Session() as session:

		# Initializer and restoring model.
		session.run(tf.global_variables_initializer())
		check = get_checkpoint(data_out_path)
		saver.restore(session, check)

		# Inputs for tensorboard.
		tf_data = tf.Variable(tf.zeros((n_images, model.z_dim)), name='tf_data')
		input_sample = tf.placeholder(tf.float32, shape=(n_images, model.z_dim))
		set_tf_data = tf.assign(tf_data, input_sample, validate_shape=False)

		if sprite:
			# Sample images.
			gen_samples, sample_z = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, n_images=n_images, show=False)
			# Generate sprite of images.
			write_sprite_image(filename=os.path.join(data_out_path, 'gen_sprite.png'), data=gen_samples)
		else:
			sample_z = np.random.uniform(low=-1., high=1., size=(n_images, model.z_dim))

		# Variable for embedding.
		saver_latent = tf.train.Saver([tf_data])
		session.run(set_tf_data, feed_dict={input_sample: sample_z})
		saver_latent.save(sess=session, save_path=os.path.join(tensorboard_path, 'tf_data.ckpt'))

		# Tensorflow embedding.
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = tf_data.name
		if sprite:
			embedding.metadata_path = os.path.join(data_out_path, 'metadata.tsv')
			embedding.sprite.image_path = os.path.join(data_out_path, 'gen_sprite.png')
		embedding.sprite.single_image_dim.extend([model.image_height, model.image_width])
		projector.visualize_embeddings(tf.summary.FileWriter(tensorboard_path), config)	


