import tensorflow as tf

def optimizer(beta_1, loss_gen, loss_dis, loss_type, learning_rate_input_g, learning_rate_input_d, beta_2=None, clipping=None, display=True):
    trainable_variables = tf.trainable_variables()
    generator_variables = [variable for variable in trainable_variables if variable.name.startswith('generator')]
    discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith('discriminator')]

    # Optimizer variable to track with optimizer is actually used.
    optimizer_print = ''

    # Handling Batch Normalization.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if 'wasserstein distance' in loss_type and 'gradient penalty' in loss_type:
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            train_generator = tf.train.AdamOptimizer(learning_rate_input_g, beta_1, beta_2).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += 'Wasserstein Distance Gradient penalty - AdamOptimizer'
        #TODO Fix this for RMSProp.
        elif 'wasserstein distance' in loss_type and 'gradient penalty' not in loss_type:
            # Weight Clipping on Discriminator, this is done to ensure the Lipschitz constrain.
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            dis_weight_clipping = [value.assign(tf.clip_by_value(value, -clipping, clipping)) for value in discriminator_variables]
            train_discriminator = tf.group(*[train_discriminator, dis_weight_clipping])

            train_generator = tf.train.AdamOptimizer(learning_rate_input_g, beta_1, beta_2).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += 'Wasserstein Distance - AdamOptimizer'

            '''
            RMS_optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_discriminator = RMS_optimizer_dis.minimize(loss_dis , var_list=discriminator_variables)
            
            # Weight Clipping on Discriminator, this is done to ensure the Lipschitz constrain.
            dis_weight_clipping = [value.assign(tf.clip_by_value(value, -c, c)) for value in discriminator_variables]
            
            train_discriminator = tf.group(*[train_discriminator, dis_weight_clipping])
            
            # Generator.
            RMS_optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_generator = RMS_optimizer_gen.minimize(loss_gen, var_list=generator_variables)
            optimizer_print += 'Wassertein Distance - RMSPropOptimizer'
            '''

        elif loss_type == 'standard' or loss_type == 'least square' or 'relativistic' in loss_type:
            train_discriminator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_dis, var_list=discriminator_variables) 
            train_generator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_g, beta1=beta_1).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        else:
            print('Loss %s not defined' % loss_type)
            exit(1)

        if display:
            print('Optimizer: %s' % optimizer_print)
            
    return train_discriminator, train_generator
