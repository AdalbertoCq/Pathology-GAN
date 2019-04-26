import os
import tensorflow as tf
import numpy as np
import h5py
import tensorflow.contrib.gan as tfgan
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras import backend as K
from models.generative.utils import *


# Method to generate random samples from a model, it also dumps a sprite image width them.
def generate_samples(model, data, data_out_path, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'data_model_output')
	path = os.path.join(path, 'Evaluation')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % tuple(data.test.shape[1:])
	checkpoint_path = os.path.join(data_out_path, 'data_model_output')
	checkpoint_path = os.path.join(checkpoint_path, model.model_name)
	checkpoint_path = os.path.join(checkpoint_path, res)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	if not os.path.isfile(hdf5_path):
		# H5 File.
		img_shape = [num_samples] + data.test.shape[1:]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:
			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			check = get_checkpoint(checkpoint_path)
			saver.restore(session, check)

			ind = 0
			while ind < num_samples:
				gen_samples, _ = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, n_images=batches, show=False)
				a = list(range(batches))
				random.shuffle(a)
				for ran in a[:20]:
					if ind == num_samples:
						break
					storage[ind] = gen_samples[ran, :, :, :]
					plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_samples[ran, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path


def real_samples(data, data_output_path, num_samples=5000):
	path = os.path.join(data_output_path, 'data_model_output')
	path = os.path.join(path, 'Evaluation')
	path = os.path.join(path, 'Real')
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
		ind = 0
		# Training hdf5
		for batch_images, batch_labels in data.training:
			if ind == num_samples:
				break
			ran = random.choice(list(range(batch_size)))
			train_storage[ind] = batch_images[ran, :, :, :]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), batch_images[ran, :, :, :])
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
		ind = 0
		# Test hdf5 
		for batch_images, batch_labels in data.test:
			if ind == num_samples:
				break
			ran = random.choice(list(range(batch_size)))
			test_storage[ind] = batch_images[ran, :, :, :]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), batch_images[ran, :, :, :])
			ind += 1
		print('\tNumber of samples:', ind)

	return hdf5_path_train, hdf5_path_test


def real_samples_contaminated(data1, data2, percent_data1, data_output_path, num_samples=5000):
	path = os.path.join(data_output_path, 'data_model_output')
	path = os.path.join(path, 'Evaluation')
	path = os.path.join(path, 'Real')

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
		ind = 0

		limit_data1 = int(num_samples*percent_data1*1e-2)
		# Training hdf5
		for batch_images, batch_labels in data1.training:
			if ind == limit_data1:
				break
			ran = random.choice(list(range(batch_size)))
			train_storage[ind] = batch_images[ran, :, :, :]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), batch_images[ran, :, :, :])
			ind += 1
		ind1 = ind
		print('\tNumber of samples Data1:', ind1)
		for batch_images, batch_labels in data2.training:
			if ind == num_samples:
				break
			ran = random.choice(list(range(batch_size)))
			train_storage[ind] = batch_images[ran, :, :, :]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), batch_images[ran, :, :, :])
			ind += 1
		print('\tNumber of samples Data2:', ind-ind1)

	if os.path.isfile(hdf5_path_test):
		print('H5 File Image Test already created.')
		print('\tFile:', hdf5_path_test)
	else:
		hdf5_img_test_real_file = h5py.File(hdf5_path_test, mode='w')
		test_storage = hdf5_img_test_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)

		print('H5 File Image Test')
		print('\tFile:', hdf5_path_test)
		ind = 0
		# Test hdf5 
		for batch_images, batch_labels in data1.test:
			if ind == limit_data1:
				break
			ran = random.choice(list(range(batch_size)))
			test_storage[ind] = batch_images[ran, :, :, :]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), batch_images[ran, :, :, :])
			ind += 1
		ind1 = ind
		print('\tNumber of samples Data1:', ind1)
		# Test hdf5 
		for batch_images, batch_labels in data2.test:
			if ind == num_samples:
				break
			ran = random.choice(list(range(batch_size)))
			test_storage[ind] = batch_images[ran, :, :, :]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), batch_images[ran, :, :, :])
			ind += 1
		print('\tNumber of samples Data2:', ind-ind1)

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


def inception_tf_feature_activations(hdf5s, input_shape, batch_size, checkpoint_path=None):
	images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
	images = 2*images_input
	images -= 1
	images = tf.image.resize_bilinear(images, [299, 299])
	out_incept_v3 = tfgan.eval.run_inception(images=images, output_tensor='pool_3:0')

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

			print('Starting features extraction...')
			print('\tImage File:', hdf5_path)
			ind = 0
			for batch_num in range(batches):
				batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
				activations = sess.run(out_incept_v3, {images_input: batch_images})
				features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
				ind += batch_size
			print('\tFeature File:', hdf5_feature_path)
			print('\tNumber of samples:', ind)

