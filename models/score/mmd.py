import tensorflow as tf
import numpy as np
from models.score.utils import *


def maximmum_mean_discrepancy_score(x, y, sigma=1):
	xx_d = euclidean_distance(x, x)
	yy_d = euclidean_distance(y, y)
	xy_d = euclidean_distance(x, y)

	scale = tf.reduce_mean(xx_d)
	# Gaussian kernel
	k_xx = tf.exp(-(xx_d)/(2*scale*(sigma**2)))
	k_yy = tf.exp(-(yy_d)/(2*scale*(sigma**2)))
	k_xy = tf.exp(-(xy_d)/(2*scale*(sigma**2)))

	mmd = maximum_mean_discrepancy(k_xx, k_yy, k_xy)
	return mmd