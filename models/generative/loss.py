import tensorflow as tf
import numpy as np
import sys

def losses(loss_type, output_fake, output_real, logits_fake, logits_real, real_images=None, fake_images=None, discriminator=None, gp_coeff=None, mean_c_x_fake=None, 
           logs2_c_x_fake=None, input_c=None, delta=1, display=True):

    # Variable to track which loss function is actually used.
    loss_print = ''
    if 'relativistic' in loss_type:
        logits_diff_real_fake = logits_real - tf.reduce_mean(logits_fake, axis=0, keepdims=True)
        logits_diff_fake_real = logits_fake - tf.reduce_mean(logits_real, axis=0, keepdims=True)
        loss_print += 'relativistic '

        if 'standard' in loss_type:
            # Discriminator loss.
            loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.ones_like(logits_fake)))
            loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.zeros_like(logits_fake)))
            loss_dis = loss_dis_real + loss_dis_fake

            # Generator loss.
            loss_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.ones_like(logits_fake)))
            loss_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.zeros_like(logits_fake)))
            loss_gen = loss_gen_real + loss_gen_fake
            loss_print += 'standard '

        elif 'least square' in loss_type:
            # Discriminator loss.
            loss_dis_real = tf.reduce_mean(tf.square(logits_diff_real_fake-1.0))
            loss_dis_fake = tf.reduce_mean(tf.square(logits_diff_fake_real+1.0))
            loss_dis = loss_dis_real + loss_dis_fake

            # Generator loss.
            loss_gen_real = tf.reduce_mean(tf.square(logits_diff_fake_real-1.0))
            loss_gen_fake = tf.reduce_mean(tf.square(logits_diff_real_fake+1.0))
            loss_gen = loss_gen_real + loss_gen_fake
            loss_print += 'least square '

        elif 'gradient penalty' in loss_type:
            # Calculating X hat.
            epsilon = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
            x_gp = real_images*(1-epsilon) + fake_images*epsilon
            out = discriminator(x_gp, True)
            logits_gp = out[1]

            # Calculating Gradient Penalty.
            grad_gp = tf.gradients(logits_gp, x_gp)
            l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
            grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))

            # Discriminator loss.
            loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.ones_like(logits_fake)))
            loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.zeros_like(logits_fake)))
            loss_dis = loss_dis_real + loss_dis_fake + (gp_coeff*grad_penalty)

            # Generator loss.
            loss_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.ones_like(logits_fake)))
            loss_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.zeros_like(logits_fake)))
            loss_gen = loss_gen_real + loss_gen_fake
            loss_print += 'gradient penalty '

    elif 'standard' in loss_type:
        # Discriminator loss. Uses hinge loss on discriminator.
        loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(output_fake)))
        loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(output_fake)*0.9))
        loss_dis = loss_dis_fake + loss_dis_real

        # Generator loss.
        # This is where we implement -log[D(G(z))] instead log[1-D(G(z))].
        # Recall the implementation of cross-entropy, sign already in. 
        loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(output_fake)))
        loss_print += 'standard '

    elif 'least square' in loss_type:       
        # Discriminator loss.
        loss_dis_fake = tf.reduce_mean(tf.square(output_fake))
        loss_dis_real = tf.reduce_mean(tf.square(output_real-1.0))
        loss_dis = 0.5*(loss_dis_fake + loss_dis_real)

        # Generator loss.
        loss_gen = 0.5*tf.reduce_mean(tf.square(output_fake-1.0))
        loss_print += 'least square '

    elif 'wasserstein distance' in loss_type:
        # Discriminator loss.
        loss_dis_real = tf.reduce_mean(logits_real)
        loss_dis_fake = tf.reduce_mean(logits_fake)
        loss_dis = -loss_dis_real + loss_dis_fake
        loss_print += 'wasserstein distance '

        # Generator loss.
        loss_gen = -loss_dis_fake
        if 'gradient penalty' in loss_type:
            # Calculating X hat.
            epsilon = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
            x_gp = real_images*(1-epsilon) + fake_images*epsilon
            out = discriminator(x_gp, True)
            logits_gp = out[1]

            # Calculating Gradient Penalty.
            grad_gp = tf.gradients(logits_gp, x_gp)
            l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
            grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))
            loss_dis += (gp_coeff*grad_penalty)
            loss_print += 'gradient penalty '
        
    elif 'hinge' in loss_type:

        loss_dis_real = tf.reduce_mean(min(0, -1 + logits_real))
        loss_dis_fake = tf.reduce_mean(min(0, -1 - logits_fake))
        loss_dis = - (loss_dis_fake + loss_dis_real)

        loss_gen = - loss_dis_fake
        loss_print += 'hinge '

    else:
        print('Loss %s not defined' % loss_type)
        sys.exit(1)

    if 'infogan' in loss_type:
        epsilon = 1e-9
        c_sigma2 = tf.exp(logs2_c_x_fake)
        log_sigma2 = tf.log(c_sigma2 + epsilon)
        mean_sq_diff = (input_c - mean_c_x_fake)**2
        last = mean_sq_diff/(c_sigma2 + epsilon)
        mututal_loss = tf.reduce_mean(-0.5*(tf.log(2*np.pi)+log_sigma2+last))

        loss_dis -= delta*mututal_loss
        loss_gen -= delta*mututal_loss
        loss_print += 'infogan '

    if display:
        print('Loss: %s' % loss_print)

    if 'infogan' in loss_type:
        return loss_dis, loss_gen, mututal_loss
        
    return loss_dis, loss_gen


