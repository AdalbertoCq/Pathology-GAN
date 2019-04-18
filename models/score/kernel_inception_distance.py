import tensorflow as tf
import numpy as np
from models.score.utils import *


def kernel_inception_distance(x, y, gamma=1, coef=1, degree=3):
	k_xx = polinomial_kernel(x, x)
	k_yy = polinomial_kernel(y, y)
	k_xy = polinomial_kernel(x, y)
	kid = maximum_mean_discrepancy(k_xx, k_yy, k_xy)	
	return kid