import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from data_manipulation.utils import *
from models.score.utils import *
from models.score.frechet_inception_distance import *
from models.score.kernel_inception_distance import *
from models.score.mmd import *
from models.score.k_nearest_neighbor import *
from models.score.inception_score import *
from models.score.mode_score import *


class Scores(object):
	def __init__(self, hdf5_features_x, hdf5_features_y, name_x, name_y, k=1, GPU=False, display=False):
		super(Scores, self).__init__()
		self.hdf5_features_x_path = hdf5_features_x
		self.hdf5_features_y_path = hdf5_features_y
		self.hdf5_images_x_path = hdf5_features_x.replace('_features_', '_images_')
		self.hdf5_images_y_path = hdf5_features_y.replace('_features_', '_images_')
		self.name_x = name_x
		self.name_y = name_y
		self.title = '%s Features - %s Features' % (self.name_x, self.name_y)
		self.k = k

		self.fid = None
		self.kid = None
		self.mmd = None
		self.knn_x = None
		self.knn_y = None
		self.knn = None

		self.display = display

		if GPU:
			self.config = tf.ConfigProto()
		else:
			self.config = tf.ConfigProto(device_count = {'GPU': 0})
			
		self.read_hdfs()
		if self.display:
			print('Running:', self.title)
			print('Loded HDF5 Files')
			print(self.name_x, 'Shape:', self.features_x.shape)
			print(self.name_y, 'Shape:', self.features_y.shape)
		self.build_graph()
		if self.display:
			print('Created Graph.')

	def read_hdfs(self):
		self.features_x = read_hdf5(self.hdf5_features_x_path, 'features')
		self.features_y = read_hdf5(self.hdf5_features_y_path, 'features')
		self.images_x = read_hdf5(self.hdf5_images_x_path, 'images')
		self.images_y = read_hdf5(self.hdf5_images_y_path, 'images')

	def build_graph(self):
		self.x_input = tf.placeholder(dtype=tf.float32, shape=(None, self.features_x.shape[-1]), name='x_features')
		self.y_input = tf.placeholder(dtype=tf.float32, shape=(None, self.features_y.shape[-1]), name='y_features')
		self.fid_output = tfgan.eval.frechet_classifier_distance_from_activations(self.x_input, self.y_input)
		self.kid_output = kernel_inception_distance(self.x_input, self.y_input)
		self.mmd_output = maximmum_mean_discrepancy_score(self.x_input, self.y_input)
		self.indices_output, self.labels_output = k_nearest_neighbor_tf_part(self.x_input, self.y_input, k=self.k)

	def run_mmd(self):
		with tf.Session(config=self.config) as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {self.x_input:self.features_x, self.y_input:self.features_y}
			self.mmd = sess.run([self.mmd_output], feed_dict)

	def run_scores(self):
		with tf.Session(config=self.config) as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {self.x_input:self.features_x, self.y_input:self.features_y}
			self.fid, self.kid, self.mmd, self.indices, self.labels = sess.run([self.fid_output, self.kid_output, self.mmd_output, self.indices_output, self.labels_output], feed_dict)
			self.knn_x, self.knn_y,self.knn = k_nearest_neighbor_np_part(self.indices, self.labels, k=self.k, x_samples=self.features_x.shape[0])
		if self.display:
			self.report_scores()
			print()

	def report_scores(self):
		print()
		print('--------------------------------------------------------')
		print(self.title)
		print('Frechet Inception Distance:', self.fid)
		# print('Kernel Inception Distance:', self.kid)
		# print('Mean Minimum Distance:', self.mmd)
		# print('%s-NN %s Accuracy:' % (self.k, self.name_x), self.knn_x)
		# print('%s-NN %s Accuracy:' % (self.k, self.name_y), self.knn_y)
		# print('%s-NN Accuracy:' % (self.k), self.knn)
		print()
		print('--------------------------------------------------------')
		print()
		