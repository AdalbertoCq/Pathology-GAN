import tensorflow as tf
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sp_linalg


# Using tf.gradients, but it's too costly to run. Using numerical_jacobian instead.
def get_gen_jacobian(session, model, z_batch):
	# Currently tf.gradients(ys, xs): does sum(dy/dx) for all y in ys

	batch, height, width, channels = model.fake_images.shape.as_list()
	jacobian_matrix = np.zeros((height, width, channels, model.z_dim), dtype=np.float32)

	print('Starting Jacobian Calculation')
	for h in range(height):
		for w in range(width):
			for c in range(channels):
				# We iterate over the M elements of the output vector
				print('tf.gradients')
				grad_func = tf.gradients(ys=model.fake_images[:, h, w, c], xs=model.z_input)
				gradients = session.run(grad_func, feed_dict={model.z_input: z_batch})
				gradients_avg = np.reshape(np.mean(gradients[0], axis=0), (1,-1))
				jacobian_matrix[h, w, c, :] = gradients_avg

	print('Done Jacobian')

	jacobian_matrix = np.reshape(jacobian_matrix, (-1, model.z_dim))
	return jacobian_matrix


def numerical_jacobian(session, model, z_batch, epsilon = 1e-3):
    batch_size, height, width, channels = model.fake_images.shape.as_list()
    batch_size, z_dim = z_batch.shape
    
    numerical_jacobian = np.zeros((z_dim, height, width, channels), dtype=np.float32)
    im = session.run(model.fake_images, feed_dict={model.z_input: z_batch})

    for zi in range(z_dim):
        zi_batch_sample = np.array(z_batch, copy=True)
        zi_batch_sample[:, zi] += epsilon
        im_sample = session.run(model.fake_images, feed_dict={model.z_input: zi_batch_sample})
        ep = (im_sample - im)/epsilon
        numerical_jacobian[zi, :, :, :] = np.mean(ep, axis=0)
        
    numerical_jacobian = np.reshape(numerical_jacobian, (model.z_dim, -1))
    # numerical_jacobian = numerical_jacobian.T

    return numerical_jacobian


def jacobian_singular_values(session, model, z_batch):
	jacobian_matrix = numerical_jacobian(session, model, z_batch)
	jac_max_sign = matrix_singular_values(matrix=jacobian_matrix, n_sing=1, mode='LM')
	jac_min_sign = matrix_singular_values(matrix=jacobian_matrix, n_sing=1, mode='SM')
	return [jac_max_sign, jac_min_sign]


def l2_normalize(vec, epsilon=1e-12):
    suma = np.sum(vec**2)
    norm = np.sqrt(suma+ epsilon)
    return vec/norm


def power_iteration_method(matrix, power_iterations=10):
    filter_shape = matrix.shape
    filter_reshape = np.reshape(matrix, [-1, filter_shape[-1]])

    u_shape = (1, filter_shape[-1])
    # If I put trainable = False, I don't need to use tf.stop_gradient()
    u = np.random.normal(size=u_shape)

    u_norm = u
    v_norm = None

    for i in range(power_iterations):
        v_iter = np.matmul(u_norm, filter_reshape.T)
        v_norm = l2_normalize(vec=v_iter, epsilon=1e-12)
        u_iter = np.matmul(v_norm, filter_reshape)
        u_norm = l2_normalize(vec=u_iter, epsilon=1e-12)

    singular_w = np.matmul(np.matmul(v_norm, filter_reshape), u_norm.T)[0,0]

    return singular_w

# Singular value desposition with Alrnoldi Iteration Method.
def matrix_singular_values(matrix, n_sing, mode='LM'):

    filter_shape = matrix.shape
    matrix_reshape = np.reshape(matrix, [-1, filter_shape[-1]])

    #Semi-positive definite matrix A*A.T, A.T*A
    dim1, dim2 = matrix_reshape.shape
    if dim1 > dim2:
        aa_t = np.matmul(matrix_reshape.T, matrix_reshape)
    else:
        aa_t = np.matmul(matrix_reshape, matrix_reshape.T)

    
    # RuntimeWarning
    # Trows warning to use eig instead if the say too is small.1
    # Eigs is an approximation, Eig calculated all eigenvalues, and eigenvectors of the matrix.

    try:
        eigenvalues, eigenvectors = sp_linalg.eigs(A=aa_t, k=n_sing, which=mode)    
    except RuntimeWarning:
        pass
    except RuntimeError:
        eigenvalues = None
        pass

    if eigenvalues is None:
        return None
        
    if n_sing > 1:
        eigenvalues = np.sort(eigenvalues)
        if 'LM' in mode:
            eigenvalues = eigenvalues[::-1]

    sing_matrix = np.sqrt(eigenvalues)

    return sing_matrix


def filter_singular_values(model, n_sing):
	gen_filters = model.gen_filters
	dis_filters = model.dis_filters

	gen_singular = dict()
	dis_singular = dict()
	for filter in gen_filters:
		f_name = str(filter.name.split(':')[0].replace('/', '_'))
		gen_singular[f_name] = matrix_singular_values(matrix=filter.eval(), n_sing=n_sing)
	for filter in dis_filters:
		f_name = str(filter.name.split(':')[0].replace('/', '_'))
		dis_singular[f_name] = matrix_singular_values(matrix=filter.eval(), n_sing=n_sing)

	return gen_singular, dis_singular


def numerical_hessian(session, model, z_batch, epsilon=1e-3):
    batch_size, height, width, channels = model.fake_images.shape.as_list()
    batch_size, z_dim = z_batch.shape
    
    numerical_jacobian = np.zeros((batch_size, z_dim, height, width, channels), dtype=np.float32)
    
    for sample in range(batch_size):
        z_batch_sample = np.zeros((z_dim, z_dim), dtype=np.float32)
        z_batch_sample_ep = np.zeros((z_dim, z_dim), dtype=np.float32)
        for z in range(z_dim):
            z_batch_sample[z, :] = np.array(z_batch[sample, :], copy=True)
            z_batch_sample_ep[z, :] = np.array(z_batch[sample, :], copy=True)
            z_batch_sample_ep[z, z] += epsilon
        im_sample = session.run(model.fake_images, feed_dict={model.z_input: z_batch_sample})
        im_sample_ep = session.run(model.fake_images, feed_dict={model.z_input: z_batch_sample_ep})
        numerical_jacobian_sample = (im_sample_ep-im_sample)/epsilon
        numerical_jacobian[sample, :, :, :, :] = numerical_jacobian_sample
    
    numerical_jacobian = np.mean(numerical_jacobian, axis=0)
    numerical_jacobian = np.reshape(numerical_jacobian, (model.z_dim, -1))
    return numerical_jacobian