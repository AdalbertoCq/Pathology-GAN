import tensorflow as tf
import numpy as np
import sys
from models.generative.activations import *

def losses(loss_type, output_fake, output_real, logits_fake, logits_real, real_images=None, fake_images=None, discriminator=None, init=None, gp_coeff=None, mean_c_x_fake=None, 
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
            out = discriminator(x_gp, True, init=init)
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
            out = discriminator(x_gp, True, init=init)
            logits_gp = out[1]

            # Calculating Gradient Penalty.
            grad_gp = tf.gradients(logits_gp, x_gp)
            l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
            grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))
            loss_dis += (gp_coeff*grad_penalty)
            loss_print += 'gradient penalty '
        
    elif 'hinge' in loss_type:
        loss_dis_real = tf.reduce_mean(tf.maximum(tf.zeros_like(logits_real), tf.ones_like(logits_real) - logits_real))
        loss_dis_fake = tf.reduce_mean(tf.maximum(tf.zeros_like(logits_real), tf.ones_like(logits_real) + logits_fake))
        loss_dis = loss_dis_fake + loss_dis_real

        '''
        tf.reduce_mean(- tf.minimum(0., -1.0 + real_logits))
        tf.reduce_mean(  tf.maximum(0.,  1.0 - logits_real))
        
        tf.reduce_mean(- tf.minimum(0., -1.0 - fake_logits))
        tf.reduce_mean(  tf.maximum(0.,  1.0 + logits_fake))
        '''

        
        loss_gen = -tf.reduce_mean(logits_fake)
        loss_print += 'hinge '

    else:
        print('Loss: Loss %s not defined' % loss_type)
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
        print('[Loss] Loss %s' % loss_print)

    if 'infogan' in loss_type:
        return loss_dis, loss_gen, mututal_loss
        
    return loss_dis, loss_gen


def vae_loss(mean_z_xi, logs2_z_xi, vae_downsample, lr_mean_xi_z, lr_logs2_xi_z):
    # VAE Loss.
    e=1
    # KL Divergence.
    z_s2 = tf.exp(logs2_z_xi)
    z_log_s2 = tf.log(z_s2 + e)
    mean_z2 = tf.square(mean_z_xi)
    kl_divergence = -.5*tf.reduce_sum(1 + z_log_s2 - mean_z2 - z_s2, axis=-1)
    loss_prior = -tf.reduce_mean( -kl_divergence, axis=-1)

    # Reconstruction error from enconding space datapoint.
    x_s2 = tf.exp(lr_logs2_xi_z)
    exp_log_s2 = tf.square(vae_downsample - lr_mean_xi_z)/(e + x_s2)
    se = tf.log(2*np.pi) + tf.log(e + x_s2) + exp_log_s2
    sampling_expt = tf.reduce_sum(-.5*se, axis=[1,2,3])
    loss_dist_likel = -tf.reduce_mean(sampling_expt, axis=[-1])
    return loss_prior, loss_dist_likel

def loss_rasgan(logits_fake, logits_real):

    logits_diff_real_fake = logits_real - tf.reduce_mean(logits_fake, axis=0, keepdims=True)
    logits_diff_fake_real = logits_fake - tf.reduce_mean(logits_real, axis=0, keepdims=True)   

    log_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.ones_like(logits_fake))
    log_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.zeros_like(logits_fake))

    log_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.ones_like(logits_fake))
    log_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.zeros_like(logits_fake))

    # Discriminator loss.
    loss_dis_real = tf.reduce_mean(log_real_dis)
    loss_dis_fake = tf.reduce_mean(log_fake_dis)
    loss_dis = loss_dis_real + loss_dis_fake

    # Generator loss.
    loss_gen_real = tf.reduce_mean(log_fake_gen)
    loss_gen_fake = tf.reduce_mean(log_real_gen)
    loss_gen = loss_gen_real + loss_gen_fake
    
    return loss_dis, loss_gen

def gradient_penalty(real_images, fake_images, real_encoding, fake_encoding, discriminator, init):
    # # Calculating X hat.
    epsilon_I = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
    epsilon_Z = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
    x_gp = real_images*(1-epsilon_I) + fake_images*epsilon_I
    z_gp = real_encoding*(1-epsilon_Z) + fake_encoding*epsilon_Z
    output, s_x_gp, s_z_gp, s_xz_gp = discriminator(images=x_gp, encoding=z_gp, reuse=True, init=init)
    
    # # Calculating Gradient Penalty.
    grad_x_gp = tf.gradients(s_x_gp, x_gp)
    grad_z_gp = tf.gradients(s_z_gp, z_gp)
    l2_grad_x_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_x_gp), axis=[1, 2, 3]))
    l2_grad_z_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_z_gp), axis=[1]))
    grad_penalty_I = tf.reduce_sum(tf.square(l2_grad_x_gp-1.0))
    grad_penalty_Z = tf.reduce_sum(tf.square(l2_grad_z_gp-1.0))

    grad_xz_x_gp = tf.gradients(s_xz_gp, x_gp)
    grad_xz_z_gp = tf.gradients(s_xz_gp, z_gp)
    l2_grad_xz_x_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_xz_x_gp), axis=[1, 2, 3]))
    l2_grad_xz_z_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_xz_z_gp), axis=[1]))
    grad_penalty_IZ = tf.reduce_sum(tf.square(l2_grad_xz_x_gp-1.0)) + tf.reduce_sum(tf.square(l2_grad_xz_z_gp-1.0))
    
    grad_penalty_I = 0
    grad_penalty_Z = 0
    
    return grad_penalty_I, grad_penalty_Z, grad_penalty_IZ


# Relativistic Discriminator loss for BigBiGAN.
def loss_RaBigBiGAN(score_real, score_fake, real_images=None, fake_images=None, real_encoding=None, fake_encoding=None, discriminator=None, init=None, gp_coeff=None, display=True):

    loss_print = 'relativistic '
    # Scores for real and fake images and encodings.
    s_x_real, s_z_real, s_xz_real = score_real
    s_x_fake, s_z_fake, s_xz_fake = score_fake

    loss_dis_s_x, loss_gen_s_x = loss_rasgan(logits_fake=s_x_fake, logits_real=s_x_real)
    loss_dis_s_z, loss_gen_s_z = loss_rasgan(logits_fake=s_z_fake, logits_real=s_z_real)
    loss_dis_s_xz, loss_gen_s_xz = loss_rasgan(logits_fake=s_xz_fake, logits_real=s_xz_real)

    grad_penalty_I, grad_penalty_Z, grad_penalty_IZ = gradient_penalty(real_images=real_images, fake_images=fake_images, real_encoding=real_encoding, fake_encoding=fake_encoding, discriminator=discriminator, init=init)

    # Combine loss.
    loss_dis = loss_dis_s_x + loss_dis_s_z + loss_dis_s_xz + (gp_coeff*grad_penalty_I) + (gp_coeff*grad_penalty_Z) + (gp_coeff*grad_penalty_IZ)
    loss_gen = loss_gen_s_x + loss_gen_s_z + loss_gen_s_xz
    
    if display:
        print('[Loss] Loss %s' % loss_print)
        
    return loss_dis, loss_gen, loss_dis_s_x, loss_gen_s_x, loss_dis_s_z, loss_gen_s_z, loss_dis_s_xz, loss_gen_s_xz