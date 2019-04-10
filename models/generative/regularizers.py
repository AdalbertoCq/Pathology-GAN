import tensorflow as tf

def orthogonal_reg(scale):

	def ortho_reg(w):

		if len(w.shape.as_list()) > 2:
			filter_size, filter_size, input_channels, output_channels = w.shape.as_list()
			w_reshape = tf.reshape(w, (-1, output_channels))
			dim = output_channels
		else:
			output_dim, input_dim = w.shape.as_list()
			dim = input_dim
			w_reshape = w

		identity = tf.eye(dim)

		wt_w = tf.matmul(a=w_reshape, b=w_reshape, transpose_a=True)
		term = tf.multiply(wt_w, (tf.ones_like(identity)-identity))

		reg = 2*tf.nn.l2_loss(term)

		return scale*reg

	return ortho_reg