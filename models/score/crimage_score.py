
import tensorflow.contrib.gan as tfgan
from data_manipulation.utils import *
from models.score.utils import *
from models.score.frechet_inception_distance import *
from models.score.kernel_inception_distance import *
from models.score.mmd import *
from models.score.k_nearest_neighbor import *
from models.score.inception_score import *
from models.score.mode_score import *


class CRImage_Scores(object):
	def __init__(self, ref1_crimage, ref2_crimage, name_x, name_y, k=1, GPU=False, display=False):
		# super(Scores, self).__init__()
		self.ref1_crimage = ref1_crimage
		self.ref2_crimage = ref2_crimage
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
			
		if self.display:
			print('Running:', self.title)
			print('Loded CRImage Files')
			print(self.name_x, 'File:', self.ref1_crimage)
			print(self.name_y, 'Shape:', self.ref2_crimage)
		self.build_graph()
		if self.display:
			print('Created Graph.')

	def build_graph(self):
		self.x_input = tf.placeholder(dtype=tf.float32, shape=(None, 3), name='x_features')
		self.y_input = tf.placeholder(dtype=tf.float32, shape=(None, 3), name='y_features')
		self.fid_output = tfgan.eval.frechet_classifier_distance_from_activations(self.x_input, self.y_input)
		self.kid_output = kernel_inception_distance(self.x_input, self.y_input)
		self.mmd_output = maximmum_mean_discrepancy_score(self.x_input, self.y_input)
		self.indices_output, self.labels_output = k_nearest_neighbor_tf_part(self.x_input, self.y_input, k=self.k)

	def read_crimage(self, file_path):
	    imgs = list()
	    with open(file_path) as content:
	        for line in content:
	            line = line.replace('\n', '')
	            # 16 5 0.0003195399 0.7619048 50072
	            values = line.split(' ')
	            values.pop()
	            if len(values) != 3:
	            	values.pop()
	            if '' == line:
	                continue
	            imgs.append(values)
	    return np.array(imgs)

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

	def run_crimage_scores(self):
		score_dict = dict()
		features_x = self.read_crimage(self.ref1_crimage)
		features_y = self.read_crimage(self.ref2_crimage)

		with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {self.x_input:features_x, self.y_input:features_y}
			self.fid, self.kid, self.mmd, self.indices, self.labels = sess.run([self.fid_output, self.kid_output, self.mmd_output, self.indices_output, self.labels_output], feed_dict)
			self.knn_x, self.knn_y, self.knn = k_nearest_neighbor_np_part(self.indices, self.labels, k=self.k, x_samples=features_x.shape[0])
		
		if self.display:
			self.report_scores()