import os
import tensorflow as tf
import numpy as np
import h5py
import random
import shutil
import tensorflow.contrib.gan as tfgan
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras import backend as K
from models.generative.utils import *
from data_manipulation.utils import *


# Method to generate random samples from a model, it also dumps a sprite image width them.
def generate_samples_epoch(session, model, data_shape, epoch, evaluation_path, num_samples=5000, batches=50):
	epoch_path = os.path.join(evaluation_path, 'epoch_%s' % epoch)
	check_epoch_path = os.path.join(epoch_path, 'checkpoints')
	checkpoint_path = os.path.join(evaluation_path, '../checkpoints')
	
	os.makedirs(epoch_path)
	shutil.copytree(checkpoint_path, check_epoch_path)

	if model.conditional:
		runs = ['postive', 'negative']
	else:
		runs = ['unconditional']

	for run in  runs:

		hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_%s_gen_images_%s.h5' % (epoch, run))
		
		# H5 File.
		img_shape = [num_samples] + data_shape
		hdf5_file = h5py.File(hdf5_path, mode='w')
		storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)

		ind = 0
		while ind < num_samples:
			if model.conditional:
				label_input = model.label_input
				if 'postive' in run:
					labels = np.ones((batches, 1))
				else:
					labels = np.zeros((batches, 1))
				labels = tf.keras.utils.to_categorical(y=labels, num_classes=2)
			else:
				label_input=None
				labels=None
			gen_samples, _ = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, label_input=label_input, labels=labels, n_images=batches, show=False)

			for i in range(batches):
				if ind == num_samples:
					break
				storage[ind] = gen_samples[i, :, :, :]
				ind += 1

def generate_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'evaluation')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % tuple(data.test.shape[1:])
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	for batch_images, batch_labels in data.training:
		break
	
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.test.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		if model.model_name == 'PathologyGAN':
			w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:
				# Image and latent generation for StylePathologyGAN.
				if model.model_name == 'PathologyGAN':
					z_latent_batch = np.random.normal(size=(batches, model.z_dim))
					feed_dict = {model.z_input_1: z_latent_batch, model.real_images:batch_images}
					w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
					w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])
					feed_dict = {model.w_latent_in:w_latent_in, model.real_images:batch_images}
					gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]
				
				# Image and latent generation for PathologyGAN.
				elif model.model_name == 'BigGAN':
					z_latent_batch = np.random.normal(size=(batches, model.z_dim))
					feed_dict = {model.z_input:z_latent_batch, model.real_images:batch_images}
					gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					img_storage[ind] = gen_img_batch[i, :, :, :]
					z_storage[ind] = z_latent_batch[i, :]
					if model.model_name == 'PathologyGAN':
						w_storage[ind] = w_latent_batch[i, :]
					plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path


def real_samples(data, data_output_path, num_samples=5000):
	path = os.path.join(data_output_path, 'evaluation')
	path = os.path.join(path, 'real')
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % (data.training.patch_h, data.training.patch_w, data.training.n_channels)
	path = os.path.join(path, res)
	img_train = os.path.join(path, 'img_train')
	img_test = os.path.join(path, 'img_test')
	if not os.path.isdir(path):
		os.makedirs(path)
		os.makedirs(img_train)
		os.makedirs(img_test)

	batch_size = data.training.batch_size
	images_shape =  [num_samples] + data.test.shape[1:]

	hdf5_path_train = os.path.join(path, 'hdf5_%s_%s_images_train_real.h5' % (data.dataset, data.marker))
	hdf5_path_test = os.path.join(path, 'hdf5_%s_%s_images_test_real.h5' % (data.dataset, data.marker))
	
	if os.path.isfile(hdf5_path_train):
		print('H5 File Image Train already created.')
		print('\tFile:', hdf5_path_train)
	else:
		hdf5_img_train_real_file = h5py.File(hdf5_path_train, mode='w')
		train_storage = hdf5_img_train_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
		
		print('H5 File Image Train.')
		print('\tFile:', hdf5_path_test)

		possible_samples = len(data.training.images)
		random_samples = list(range(possible_samples))
		random.shuffle(random_samples)

		ind = 0
		for index in random_samples[:num_samples]:
			train_storage[ind] = data.training.images[index]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), data.training.images[index])
			ind += 1
		print('\tNumber of samples:', ind)

	if os.path.isfile(hdf5_path_test):
		print('H5 File Image Test already created.')
		print('\tFile:', hdf5_path_test)
	else:
		hdf5_img_test_real_file = h5py.File(hdf5_path_test, mode='w')
		test_storage = hdf5_img_test_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)

		print('H5 File Image Test')
		print('\tFile:', hdf5_path_test)

		possible_samples = len(data.test.images)
		random_samples = list(range(possible_samples))
		random.shuffle(random_samples)

		ind = 0
		for index in random_samples[:num_samples]:
			test_storage[ind] = data.test.images[index]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), data.test.images[index])
			ind += 1
		print('\tNumber of samples:', ind)

	return hdf5_path_train, hdf5_path_test


def real_samples_cond(data, cond, data_output_path, num_samples=5000):
	path = os.path.join(data_output_path, 'evaluation')
	path = os.path.join(path, 'real')
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % (data.training.patch_h, data.training.patch_w, data.training.n_channels)
	path = os.path.join(path, res)

	if cond == 1:
		name_1 = 'er_positive'
		name_2 = 'er_negative'
	elif cond == 0:
		name_1 = 'survival_positive'
		name_2 = 'survival_negative'

	er_p_path = os.path.join(path, name_1)
	er_n_path = os.path.join(path, name_2)

	er_p_img_train = os.path.join(er_p_path, 'img_train')
	er_p_img_test = os.path.join(er_p_path, 'img_test')
	if not os.path.isdir(er_p_path):
		os.makedirs(er_p_path)
		os.makedirs(er_p_img_train)
		os.makedirs(er_p_img_test)

	hdf5_path_er_p_train = os.path.join(er_p_path, 'hdf5_%s_%s_images_train_real.h5' % (data.dataset, data.marker))
	hdf5_path_er_p_test = os.path.join(er_p_path, 'hdf5_%s_%s_images_test_real.h5' % (data.dataset, data.marker))
	

	er_n_img_train = os.path.join(er_n_path, 'img_train')
	er_n_img_test = os.path.join(er_n_path, 'img_test')
	if not os.path.isdir(er_n_path):
		os.makedirs(er_n_path)
		os.makedirs(er_n_img_train)
		os.makedirs(er_n_img_test)

	hdf5_path_er_n_train = os.path.join(er_n_path, 'hdf5_%s_%s_images_train_real.h5' % (data.dataset, data.marker))
	hdf5_path_er_n_test = os.path.join(er_n_path, 'hdf5_%s_%s_images_test_real.h5' % (data.dataset, data.marker))
	
	batch_size = data.training.batch_size
	images_shape =  [num_samples] + data.test.shape[1:]

	if os.path.isfile(hdf5_path_er_p_train) and os.path.isfile(hdf5_path_er_n_train):
		print('H5 File Image Train already created.')
		print('\tFile:', hdf5_path_er_p_train)
		print('\tFile:', hdf5_path_er_n_train)
	else:
		hdf5_er_p_img_train_real_file = h5py.File(hdf5_path_er_p_train, mode='w')
		hdf5_er_n_img_train_real_file = h5py.File(hdf5_path_er_n_train, mode='w')
		er_p_train_storage = hdf5_er_p_img_train_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
		er_n_train_storage = hdf5_er_n_img_train_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
		
		print('H5 File Image Train.')
		print('\tFile:', hdf5_path_er_p_train)
		print('\tFile:', hdf5_path_er_n_train)

		possible_samples = len(data.training.images)
		random_samples = list(range(possible_samples))
		random.shuffle(random_samples)

		ind_er_p = 0
		ind_er_n = 0
		for index in random_samples:
			label =  data.training.labels[index][cond]
			if cond == 0:
				label = survival_5(label)
			if label == 1. and ind_er_p < num_samples:
				er_p_train_storage[ind_er_p] = data.training.images[index]	
				plt.imsave('%s/real_train_%s.png' % (er_p_img_train, ind_er_p), data.training.images[index])
				ind_er_p += 1
			elif label == 0. and ind_er_n < num_samples:
				er_n_train_storage[ind_er_n] = data.training.images[index]
				plt.imsave('%s/real_train_%s.png' % (er_n_img_train, ind_er_n), data.training.images[index])
				ind_er_n += 1
			elif ind_er_p == num_samples-1 and ind_er_n == num_samples-1:
				break
			
		print('\tNumber of samples %s:' % name_1, ind_er_p)
		print('\tNumber of samples %s:' % name_2, ind_er_n)

	if os.path.isfile(hdf5_path_er_p_test) and os.path.isfile(hdf5_path_er_n_test):
		print('H5 File Image Test already created.')
		print('\tFile:', hdf5_path_er_p_test)
		print('\tFile:', hdf5_path_er_n_test)
	else:
		hdf5_er_p_img_test_real_file = h5py.File(hdf5_path_er_p_test, mode='w')
		hdf5_er_n_img_test_real_file = h5py.File(hdf5_path_er_n_test, mode='w')
		er_p_test_storage = hdf5_er_p_img_test_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
		er_n_test_storage = hdf5_er_n_img_test_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)

		print('H5 File Image Test')
		print('\tFile:', hdf5_path_er_p_test)
		print('\tFile:', hdf5_path_er_n_test)

		possible_samples = len(data.test.images)
		random_samples = list(range(possible_samples))
		random.shuffle(random_samples)

		ind_er_p = 0
		ind_er_n = 0
		for index in random_samples:
			label =  data.training.labels[index][cond]
			if cond == 0:
				label = survival_5(label)
			if label == 1. and ind_er_p < num_samples:
				er_p_test_storage[ind_er_p] = data.test.images[index]	
				plt.imsave('%s/real_test_%s.png' % (er_p_img_test, ind_er_p), data.test.images[index])
				ind_er_p += 1
			elif label == 0. and ind_er_n < num_samples:
				er_n_test_storage[ind_er_n] = data.test.images[index]
				plt.imsave('%s/real_test_%s.png' % (er_n_img_test, ind_er_n), data.test.images[index])
				ind_er_n += 1
			elif ind_er_p == num_samples-1 and ind_er_n == num_samples-1:
				break

		print('\tNumber of samples %s:' % name_1, ind_er_p)
		print('\tNumber of samples %s:' % name_2, ind_er_n)

	return hdf5_path_er_p_train, hdf5_path_er_p_test, hdf5_path_er_n_train, hdf5_path_er_n_test


def real_samples_contaminated(data1, data2, percent_data1, data_output_path, num_samples=5000):
	path = os.path.join(data_output_path, 'evaluation')
	path = os.path.join(path, 'real')

	dataset = '%s_%s_contaminated_%sperc' % (data1.dataset, data2.dataset, percent_data1)
	path = os.path.join(path, dataset)
	marker = '%s_%s_contaminated_%sperc' % (data1.marker, data2.marker, percent_data1)
	path = os.path.join(path, marker)
	res = 'h%s_w%s_n%s' % (data1.training.patch_h, data1.training.patch_w, data1.training.n_channels)
	
	path = os.path.join(path, res)
	img_train = os.path.join(path, 'img_train')
	img_test = os.path.join(path, 'img_test')
	if not os.path.isdir(path):
		os.makedirs(path)
		os.makedirs(img_train)
		os.makedirs(img_test)

	batch_size = data1.training.batch_size
	images_shape =  [num_samples] + data1.test.shape[1:]

	hdf5_path_train = os.path.join(path, 'hdf5_%s_%s_%sperc_images_train_real.h5' % (dataset, marker, percent_data1))
	hdf5_path_test = os.path.join(path, 'hdf5_%s_%s_%sperc_images_test_real.h5' % (dataset, marker, percent_data1))
	
	if os.path.isfile(hdf5_path_train):
		print('H5 File Image Train already created.')
		print('\tFile:', hdf5_path_train)
	else:
		hdf5_img_train_real_file = h5py.File(hdf5_path_train, mode='w')
		train_storage = hdf5_img_train_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
		
		print('H5 File Image Train.')
		print('\tFile:', hdf5_path_train)

		limit_data1 = int(num_samples*percent_data1*1e-2)
		limit_data2 = num_samples - limit_data1
		
		possible_samples_1 = len(data1.training.images)
		random_samples_1 = list(range(possible_samples_1))
		random.shuffle(random_samples_1)

		possible_samples_2 = len(data2.training.images)
		random_samples_2 = list(range(possible_samples_2))
		random.shuffle(random_samples_2)

		ind = 0
		for index in random_samples_1[:limit_data1]:
			train_storage[ind] = data1.training.images[index]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), data1.training.images[index])
			ind += 1
		ind1 = ind
		print('\tNumber of samples Data1:', ind)

		for index in random_samples_2[:limit_data2]:
			train_storage[ind] = data2.training.images[index]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), data2.training.images[index])
			ind += 1
		print('\tNumber of samples Data1:', ind-ind1)
		print('\tTotal Number of Samples:', ind)


	if os.path.isfile(hdf5_path_test):
		print('H5 File Image Test already created.')
		print('\tFile:', hdf5_path_test)
	else:
		hdf5_img_test_real_file = h5py.File(hdf5_path_test, mode='w')
		test_storage = hdf5_img_test_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)

		print('H5 File Image Test')
		print('\tFile:', hdf5_path_test)
		
		possible_samples_1 = len(data1.test.images)
		random_samples_1 = list(range(possible_samples_1))
		random.shuffle(random_samples_1)

		possible_samples_2 = len(data2.test.images)
		random_samples_2 = list(range(possible_samples_2))
		random.shuffle(random_samples_2)

		ind = 0
		for index in random_samples_1[:limit_data1]:
			test_storage[ind] = data1.test.images[index]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), data1.test.images[index])
			ind += 1
		ind1 = ind
		print('\tNumber of samples Data1:', ind)

		for index in random_samples_2[:limit_data2]:
			test_storage[ind] = data2.test.images[index]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), data2.test.images[index])
			ind += 1
		print('\tNumber of samples Data1:', ind-ind1)
		print('\tTotal Number of Samples:', ind)

	return hdf5_path_train, hdf5_path_test


def inception_feature_activations(hdf5s, input_shape, batch_size, checkpoint_path=None):
	
	# Inception Network.
	# Pool_3 layer output, features for FID.
	if checkpoint_path is None:
		print('InceptionV3: ImageNet pre-trained network.')
		images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
		images = 2*images_input
		images -= 1
		images = tf.image.resize_bilinear(images, [299, 299])
		inception_v3_no_top = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=[299, 299, 3], input_tensor=images)
	else:	
		print('InceptionV3: Fine-tuned trained network.')
		images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
		images = 2*images_input
		images -= 1
		images = tf.image.resize_bilinear(images, [299, 299])
		inception_v3_no_top = inception_v3.InceptionV3(include_top=False, weights=None, input_shape=[299, 299, 3], input_tensor=images)
	net = GlobalAveragePooling2D()(inception_v3_no_top.output)    
	inception_v3_model = Model(inputs=inception_v3_no_top.input, outputs=net)
	print('Pool_3 InceptionV3 Features:', net.shape.as_list()[-1])

	saver = tf.train.Saver()
	with tf.keras.backend.get_session() as sess:
		K.set_session(sess)
		sess.run(tf.global_variables_initializer())
		if checkpoint_path is not None:
			saver.restore(sess, checkpoint_path)
			print('Restored Model: %s' % checkpoint_path)

		for hdf5_path in hdf5s:
			hdf5_img_file = h5py.File(hdf5_path, mode='r')
			if 'images' not in hdf5_img_file:
				images_storage = hdf5_img_file['generated_images']
			else:
				images_storage = hdf5_img_file['images']
			num_samples = images_storage.shape[0]
			batches = int(num_samples/batch_size)

			features_shape = (num_samples, net.shape.as_list()[-1])
			hdf5_feature_path = hdf5_path.replace('_images_','_features_')
			if os.path.isfile(hdf5_feature_path):
				os.remove(hdf5_feature_path)
			hdf5_features_file = h5py.File(hdf5_feature_path, mode='w')
			features_storage = hdf5_features_file.create_dataset(name='features', shape=features_shape, dtype=np.float32)

			print('Starting features extraction...')
			print('\tImage File:', hdf5_path)
			ind = 0
			for batch_num in range(batches):
				batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
				feed_dict = {images_input:batch_images}
				activations = sess.run(inception_v3_model.outputs, feed_dict)
				features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations[0]
				ind += batch_size
			print('\tFeature File:', hdf5_feature_path)
			print('\tNumber of samples:', ind)


def inception_tf_feature_activations(hdf5s, input_shape, batch_size):
	images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
	images = 2*images_input
	images -= 1
	images = tf.image.resize_bilinear(images, [299, 299])
	out_incept_v3 = tfgan.eval.run_inception(images=images, output_tensor='pool_3:0')

	hdf5s_features = list()
	with tf.Session() as sess:
		for hdf5_path in hdf5s:
			hdf5_img_file = h5py.File(hdf5_path, mode='r')
			images_storage = hdf5_img_file['images']
			num_samples = images_storage.shape[0]
			batches = int(num_samples/batch_size)

			features_shape = (num_samples, 2048)
			hdf5_feature_path = hdf5_path.replace('_images_','_features_')
			if os.path.isfile(hdf5_feature_path):
				os.remove(hdf5_feature_path)
			hdf5_features_file = h5py.File(hdf5_feature_path, mode='w')
			features_storage = hdf5_features_file.create_dataset(name='features', shape=features_shape, dtype=np.float32)
			hdf5s_features.append(hdf5_feature_path)

			print('Starting features extraction...')
			print('\tImage File:', hdf5_path)
			ind = 0
			for batch_num in range(batches):
				batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
				if 'test' in hdf5_path or 'train' in hdf5_path:
					batch_images = batch_images/255.
				activations = sess.run(out_incept_v3, {images_input: batch_images})
				features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
				ind += batch_size
			print('\tFeature File:', hdf5_feature_path)
			print('\tNumber of samples:', ind)

	return hdf5s_features