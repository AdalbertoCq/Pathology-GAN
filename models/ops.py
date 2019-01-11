import tensorflow as tf


def _pool(func, x, size, stride):
    return func(x, [1, size, size, 1], [1, stride, stride, 1], 'SAME')


def avg_pool(x, size, stride=None):
    if stride is None:
        stride = size
    return _pool(tf.nn.avg_pool, x, size, stride)


def global_avg_pool(x):
    size = x.get_shape()[-2].value
    return tf.nn.avg_pool(x, [1, size, size, 1], [1, 1, 1, 1], padding='VALID')


def max_pool(x, size, stride=None):
    if stride is None:
        stride = size
    return _pool(tf.nn.max_pool, x, size, stride)


def lrelu(input_, leak=0.2):
    return tf.maximum(input_, leak * input_)


def upsample(x, size=2):
    _, h, w, _ = x.get_shape()
    return tf.image.resize_images(x, [size * h.value, size * w.value])


def batch_norm(x, training, momentum=0.9):
    return tf.layers.batch_normalization(x, momentum=momentum, training=training)

# Regular convolution
# size = Size of kernel.
# filters = # output channels.
# stride = Stride (default=1)
def conv(input_, size, filters, stride=1):
    channels = input_.get_shape()[-1].value
    kernel = tf.get_variable('kernel', shape=[size, size, channels, filters],
                             initializer=tf.contrib.layers.variance_scaling_initializer())
    output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], 'SAME')
    return output

# Convolutional transpose.
def conv_t(x, size, filters, stride=2):
    _, height, width, depth = list(map(lambda dim: dim.value, x.get_shape()))
    kernel = tf.get_variable('kernel', shape=[size, size, filters, depth],
                             initializer=tf.contrib.layers.variance_scaling_initializer())
    
    partial_output_shape = [stride * height, stride * width, filters]
    
    # TODO: Not clear about this. What's its purpose? Is it because it's providing the kernel?
    output_shape = tf.concat([tf.shape(x)[:1], tf.constant(partial_output_shape, dtype=tf.int32)], axis=0)
    
    output = tf.nn.conv2d_transpose(x, kernel, output_shape, strides=[1, stride, stride, 1])
    
    output.set_shape([None] + partial_output_shape)
    return output


def dense(input_, units, use_bias=True, activation=None):
    channels = input_.get_shape()[-1].value
    kernel = tf.get_variable('kernel', shape=[channels, units], initializer=tf.contrib.layers.xavier_initializer())
    output = tf.matmul(input_, kernel)
    if use_bias:
        output += tf.get_variable('bias', shape=[units], initializer=tf.zeros_initializer())
    if activation:
        output = activation(output)
    return output


def avg_pool_2(input_, **kwargs):
    assert not kwargs
    return avg_pool(input_, 2, 2)

# Convolutional transpose 5x5 and stride 2.
# If no filter number specified, the number of channels is compressed to 1/4.
def conv_t_5(input_, filters=None):
    if filters is None:
        filters = round(input_.get_shape()[-1].value / 4)
    output = conv_t(input_, 5, filters)
    return output

# Convolution for kernels of size 5
# and output channels specified in filters.
def conv_5_2(input_, filters=None):
    if filters is None:
        filters = round(input_.get_shape()[-1].value)
    output = conv(input_, 5, filters, 2)
    return output
