import tensorflow as tf

def polinomial_kernel(x, y, gamma=1., coef=1, degree=3):
	# Pair-wise dot product.
	xy = dot_product(x, y)
	g = gamma/float(x.shape.as_list()[1])
	xy_g = xy*g
	kernel = tf.pow(xy_g + coef, degree)
	return kernel


# Euclidean distance, pair-wise comparison. 
def euclidean_distance(x, y, squared=True):
	# num_sample_x = x.shape.as_list()[0]
	# num_sample_y = y.shape.as_list()[0]

	# x_reshape = tf.reshape(x, shape=(num_sample_x, -1))
	# y_reshape = tf.reshape(y, shape=(num_sample_y, -1))

	x_sum_dims = sum(x.shape.as_list()[1:])
	y_sum_dims = sum(y.shape.as_list()[1:])

	x_reshape = tf.reshape(x, shape=(-1, x_sum_dims))
	y_reshape = tf.reshape(y, shape=(-1, y_sum_dims))

	x_2 = tf.reduce_sum(x_reshape**2, axis=-1, keepdims=True)
	y_2 = tf.reduce_sum(y_reshape**2, axis=-1, keepdims=True)
	y_2 = tf.transpose(y_2)
	x_y = dot_product(x, y)
    
	distance = x_2 + y_2 - 2*x_y

	if not squared:
		# Maintain distance, removing negative values
		distance = (distance + tf.abs(distance))/2
		distance = tf.sqrt(distance)

	# print('distance', distance)

	return distance

# Dot product of all pair-wise vectors.
def dot_product(x, y):
	x_sum_dims = sum(x.shape.as_list()[1:])
	y_sum_dims = sum(y.shape.as_list()[1:])

	x_reshape = tf.reshape(x, shape=(-1, x_sum_dims))
	y_reshape = tf.reshape(y, shape=(-1, y_sum_dims))

	dot_prod = tf.matmul(x_reshape, y_reshape, transpose_b=True)
	return dot_prod

def covariance(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(mean_x, mean_x, transpose_a=True)
    vx = tf.matmul(x, x, transpose_a=True)/tf.cast(x.shape.as_list()[0], tf.float32)
    covariance = vx - mx
    return covariance

def maximum_mean_discrepancy(k_xx, k_yy, k_xy):
	samples_x = tf.cast(tf.shape(k_xx)[0], dtype=tf.float32)
	samples_y = tf.cast(tf.shape(k_yy)[0], dtype=tf.float32)

	k_xx_diag = tf.multiply(k_xx, tf.eye(tf.shape(k_xx)[0]))
	k_xx = k_xx - k_xx_diag

	k_yy_diag = tf.multiply(k_yy, tf.eye(tf.shape(k_yy)[0]))
	k_yy = k_yy - k_yy_diag

	E_xx = tf.reduce_sum(k_xx)/(samples_x*(samples_x-1))
	E_yy = tf.reduce_sum(k_yy)/(samples_y*(samples_y-1))
	E_xy = tf.reduce_mean(k_xy)
	mmd_2 = E_xx + E_yy - 2*E_xy
	mmd = tf.sqrt(tf.maximum(mmd_2,0))
	return mmd
