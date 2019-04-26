import tensorflow as tf
import numpy as np
from scipy import linalg
from models.score.utils import *


def frechet_inception_distance(x_features, y_features, batch_size, sqrt=False):
	batch_scores = list()
	batches = int(x_features.shape.as_list()[0]/batch_size)
	for i in range(batches):
		if batches-1 == i: 
			x_features_batch = x_features[i*batch_size: , :]
			y_features_batch = y_features[i*batch_size: , :]
		else:
			x_features_batch = x_features[i*batch_size : (i+1)*batch_size, :]
			y_features_batch = y_features[i*batch_size : (i+1)*batch_size, :]
		
		samples = x_features_batch.shape.as_list()[0]
		x_feat = tf.reshape(x_features_batch, (samples, -1))
		y_feat = tf.reshape(y_features_batch, (samples, -1))

		x_mean = tf.reduce_mean(x_feat, axis=0)
		y_mean = tf.reduce_mean(y_feat, axis=0)

		# Review this two lines.
		x_cov = covariance(x_feat)
		y_cov = covariance(y_feat)

		means = dot_product(x_mean, x_mean) + dot_product(y_mean, y_mean) - 2*dot_product(x_mean, y_mean)
		cov_s = linalg.sqrtm(tf.matmul(x_cov, y_cov), True)
		cov_s = cov_s.real
		covas = tf.trace(x_cov + y_cov - 2*cov_s)

		fid = means + covas
		if sqrt:
			fid = tf.sqrt(fid)
		batch_scores.append(np.array(fid))
	return np.mean(batch_scores), np.std(batch_scores)