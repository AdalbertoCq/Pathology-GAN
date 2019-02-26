import tensorflow as tf
# from models.generative.utils import power_iteration_method

def leakyReLU(x, alpha):
    return tf.maximum(alpha*x, x)

# This step need to be heavily revised.
def attention_block(x, i, spectral=True, power_iterations=1):

    batch_size, height, width, channels = x.get_shape().as_list()
    with tf.variable_scope('attention_block_%s' % i):

        # Global value for all pixels, measures how important is the context for each of them.
        gamma = tf.get_variable('gamma', shape=(1),initializer=tf.constant_initializer(0.0))
        f_g_channels = channels//8

        f = convolutional(inputs=x, output_channels=f_g_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, power_iterations=power_iterations, scope=1)
        g = convolutional(inputs=x, output_channels=f_g_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, power_iterations=power_iterations, scope=2)
        h = convolutional(inputs=x, output_channels=channels    , filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, power_iterations=power_iterations, scope=3)

        # Flatten f, g, and h per channel.
        f_flat = tf.reshape(f, shape=tf.stack([tf.shape(x)[0], height*width, channels//8]))
        g_flat = tf.reshape(g, shape=tf.stack([tf.shape(x)[0], height*width, channels//8]))
        h_flat = tf.reshape(h, shape=tf.stack([tf.shape(x)[0], height*width, channels]))

        s = tf.matmul(g_flat, f_flat, transpose_b=True)

        beta = tf.nn.softmax(s)

        print(beta)
        print(x.shape)
        print(beta.shape)
        print(h_flat.shape)
        o = tf.matmul(beta, h_flat)
        o = tf.reshape(o, shape=tf.stack([tf.shape(x)[0], height, width, channels]))
        y = gamma*o + x

    return y

def spectral_normalization(filter, power_iterations):
    # Vector is preserved after each SGD iteration, good performance with power_iter=1 and presenving. 
    # Need to make v trainable, and stop gradient descent to going through this path/variables.
    # Isotropic gaussian. 

    filter_shape = filter.get_shape()
    filter_reshape = tf.reshape(filter, [-1, filter_shape[-1]])
    
    u_shape = (1, filter_shape[-1])
    # If I put trainable = False, I don't need to use tf.stop_gradient()
    u = tf.get_variable('u', shape=u_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(), trainable=False)

    # u_norm, singular_w = power_iteration_method(filter_reshape, u, power_iterations)

    u_norm = u
    v_norm = None

    for i in range(power_iterations):
        v_iter = tf.matmul(u_norm, tf.transpose(filter_reshape))
        v_norm = tf.math.l2_normalize(x=v_iter, epsilon=1e-12)
        u_iter = tf.matmul(v_norm, filter_reshape)
        u_norm = tf.math.l2_normalize(x=u_iter, epsilon=1e-12)

    singular_w = tf.matmul(tf.matmul(v_norm, filter_reshape), tf.transpose(u_norm))[0,0]

    '''
    tf.assign(ref,  value):
        This operation outputs a Tensor that holds the new value of 'ref' after the value has been assigned. 
        This makes it easier to chain operations that need to use the reset value.
        Do the previous iteration and assign u.

    with g.control_dependencies([a, b, c]):
            `d` and `e` will only run after `a`, `b`, and `c` have executed.

    To keep value of u_nom in u?
    If I put this here, the filter won't be use in here until the normalization is done and the value of u_norm kept in u.
    The kernel of the conv it's a variable it self, with its dependencies.
    '''
    with tf.control_dependencies([u.assign(u_norm)]):
        filter_normalized = filter / singular_w
        filter_normalized = tf.reshape(filter_normalized, filter.shape)

    # We can control the normalization before the executing the optimizer by runing the update of all the assign operations 
    # in the variable collection.
    # filter_normalized = filter / singular_w
    # filter_normalized = tf.reshape(filter_normalized, filter.shape)
    # tf.add_to_collection('SPECTRAL_NORM_UPDATE_OPS', u.assign(u_norm))

    '''
    CODE TRACK SINGULAR VALUE OF WEIGHTS.
    filter_normalized_reshape = filter_reshape / singular_w
    s, _, _ = tf.svd(filter_normalized_reshape)
    tf.summary.scalar(filter.name, s[0])
    '''
    
    return filter_normalized


def convolutional(inputs, output_channels, filter_size, stride, padding, conv_type, scope, data_format='NHWC', output_shape=None, spectral=False, power_iterations=1):
    with tf.variable_scope('conv_layer_%s' % scope):
        '''
        Kernel and bias initilization, adding this whole documentation to make sure transition for tf.contrib.layer.conv2d to tf.nn.conv2d has the same setup as 
        previous models:
            tf.get_variable(
                name,
                shape=None, 
                dtype=None,
                initializer=None,
                regularizer=None,
                trainable=None,
                collections=None,
                caching_device=None,
                partitioner=None,
                validate_shape=True,
                use_resource=None,
                custom_getter=None,
                constraint=None,
                synchronization=tf.VariableSynchronization.AUTO,
                aggregation=tf.VariableAggregation.NONE
            )
        '''
        # weight_init = tf.initializers.random_normal(stddev=0.02)
        weight_init = tf.contrib.layers.xavier_initializer_conv2d()

        # Shapes.
        current_shape = inputs.get_shape()
        input_channels = current_shape[3]
        if 'transpose'in conv_type or 'upscale' in conv_type: filter_shape = (filter_size, filter_size, output_channels, input_channels)   
        else: filter_shape = (filter_size, filter_size, input_channels, output_channels)    

        bias = tf.get_variable(name='bias', shape=[output_channels], initializer=tf.constant_initializer(0.0), trainable=True, dtype=tf.float32) 
        filter = tf.get_variable(name='filter', shape=filter_shape, initializer=weight_init, trainable=True, dtype=tf.float32)    
        
        '''
        Comparison betweeen previous config to new: tf.contrib.layer.conv2d to tf.nn.conv2d has the same setup.
            USAGE: weight_init = tf.contrib.layers.xavier_initializer_conv2d()
                    tf.nn.conv2d_transpose(
                            value=inputs,
                            filter=filter,
                            output_shape=tf.stack([tf.shape(inputs)[0], current_shape[1]*stride, current_shape[2]*stride, output_channels]),
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            data_format='NHWC',
                            name=None
                    )

                    kernel_shape = self.kernel_size + (self.filters, input_dim)

                    self.add_weight(
                            name='kernel',
                            shape=kernel_shape,
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint,
                            trainable=True,
                            dtype=self.dtype,
                            partitioner=None,
                            use_resource=None,
                            synchronization=tf_variables.VariableSynchronization.AUTO,
                            aggregation=tf_variables.VariableAggregation.NONE
                    )
                            
                
                    nn.conv2d_transpose(
                            inputs,
                            self.kernel,
                            output_shape_tensor=(batch_size, out_height, out_width, self.filters),
                            stride=(1, stride_h, stride_w, 1),
                            padding=self.padding.upper(),
                            data_format=conv_utils.convert_data_format(self.data_format, ndim=4)
                    )
                    tf.layers.conv2d_transpose(
                            inputs=net,
                            filters=256,
                            kernel_size=(2,2),
                            strides=(2, 2),
                            padding='same',
                            data_format='channels_last',                                DEFAULT.
                            activation=None,                                            DEFAULT.
                            use_bias=True,                                              DEFAULT.
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                            bias_initializer=tf.zeros_initializer(),                    DEFAULT.
                            kernel_regularizer=None,                                    DEFAULT.
                            bias_regularizer=None,                                      DEFAULT.    
                            activity_regularizer=None,                                  DEFAULT.
                            kernel_constraint=None,                                     DEFAULT.
                            bias_constraint=None,                                       DEFAULT.
                            trainable=True,                                             DEFAULT.
                            name=None,                                                  DEFAULT.
                            reuse=None                                                  DEFAULT.
                    )

            USAGE:  nn.conv2d(input=inputs, filter=filter, strides=strides, padding='SAME', data_format='NHWC')
                    layers.conv2d(inputs=net, filters=128, kernel_size=(5,5), strides=(2, 2), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
            DEFAULTS: tf.layers.conv2d(
                            inputs,
                            filters,
                            kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            data_format='channels_last',
                            dilation_rate=(1, 1),
                            activation=None,
                            use_bias=True,
                            kernel_initializer=None,
                            bias_initializer=tf.zeros_initializer(),
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            trainable=True,
                            name=None,
                            reuse=None
            )
        '''
        # Type of convolutional operation.
        if conv_type == 'upscale':
            output_shape = [tf.shape(inputs)[0], current_shape[1]*2, current_shape[2]*2, output_channels]
            # Weight filter initializer.
            filter = tf.pad(filter, ([1,1], [1,1], [0,0], [0,0]), mode='CONSTANT')
            filter = tf.add_n([filter[1:,1:], filter[:-1,1:], filter[1:,:-1], filter[:-1,:-1]])
            if spectral: filter = spectral_normalization(filter, power_iterations)
            strides = [1, 2, 2, 1]
            output = tf.nn.conv2d_transpose(value=inputs, filter=filter, output_shape=tf.stack(output_shape), strides=strides, padding=padding, data_format=data_format)
            
        elif conv_type == 'downscale':
            # Weight filter initializer.
            filter = tf.pad(filter, ([1,1], [1,1], [0,0], [0,0]), mode='CONSTANT')
            filter = tf.add_n([filter[1:,1:], filter[:-1,1:], filter[1:,:-1], filter[:-1,:-1]])
            if spectral: filter = spectral_normalization(filter, power_iterations)
            strides = [1, 2, 2, 1]
            output = tf.nn.conv2d(input=inputs, filter=filter, strides=strides, padding=padding, data_format=data_format)
            
        elif conv_type == 'transpose':
            output_shape = [tf.shape(inputs)[0], current_shape[1]*stride, current_shape[2]*stride, output_channels]
            strides = [1, stride, stride, 1]
            if spectral: filter = spectral_normalization(filter, power_iterations)
            output = tf.nn.conv2d_transpose(value=inputs, filter=filter, output_shape=tf.stack(output_shape), strides=strides, padding=padding, data_format=data_format)
        
        elif conv_type == 'convolutional':
            strides = [1, stride, stride, 1]
            if spectral: filter = spectral_normalization(filter, power_iterations)
            output = tf.nn.conv2d(input=inputs, filter=filter, strides=strides, padding=padding, data_format=data_format)
        
        output = tf.nn.bias_add(output, bias, data_format=data_format)
    return output


def dense(inputs, out_dim, scope, use_bias=True, spectral=False, power_iterations=1):
    with tf.variable_scope('dense_layer_%s' % scope):
        
        in_dim = inputs.get_shape()[-1]
        weights = tf.get_variable('kernel', shape=[in_dim, out_dim], dtype=tf.float32, trainable=True)
        
        if spectral:
            output = tf.matmul(inputs, spectral_normalization(weights, power_iterations))
        else:
            output = tf.matmul(inputs, weights)
        
        if use_bias : 
            bias = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0.0), trainable=True, dtype=tf.float32)
        output = tf.add(output, bias)
    return output


def residual_block(inputs, filter_size, stride, padding, scope, is_training=True, normalization=None, use_bias=True, spectral=False, activation=None, power_iterations=1):
    channels = inputs.shape.as_list()[-1]
    with tf.variable_scope('resblock_%s' % scope):
        with tf.variable_scope('part_1'):
            # Convolutional
            net = convolutional(inputs, channels, filter_size, stride, padding, 'convolutional', scope=1, spectral=spectral, power_iterations=power_iterations)
            # Batch Normalization
            if normalization is not None: net = normalization(inputs=net, training=is_training)
            # Activation
            if activation is not None: net = activation(net)
            
        with tf.variable_scope('part_2'):
            # Convolutional
            net = convolutional(net, channels, filter_size, stride, padding, 'convolutional', scope=1, spectral=spectral, power_iterations=power_iterations)
            # Batch Normalization
            if normalization is not None: net = normalization(inputs=net, training=is_training)
            # Activation
            if activation is not None: net = activation(net)
            
        return inputs + net


            


