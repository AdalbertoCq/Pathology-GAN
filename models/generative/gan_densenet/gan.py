import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model.generator import Generator
from model.discriminator import Discriminator
from runner.runner import Runner
import runner.utils as utils


def save_image(img, a, b):
    plt.imsave('run/%s/img/%d_%d.png' % (utils.job_id(), a, b), img)


class GAN(Runner):
    def __init__(self, noise=(56 * 56,)):
        super().__init__()

        self.real = tf.placeholder(tf.float32, [self.batch_size] + self.data.training.shape[1:])
        self.z = tf.random_uniform([self.batch_size] + list(noise), -1.0, 1.0)
        self.train_phase = tf.placeholder(tf.bool)

        self.generated = Generator().build_graph(self.z, self.train_phase)
        discriminator = Discriminator()
        self.p_real = discriminator.build_graph(self.real, self.train_phase)
        self.p_fake = discriminator.build_graph(self.generated, self.train_phase, reuse=True)

        epsilon = 1e-3
        self.loss_d = -tf.reduce_mean(tf.log(epsilon + self.p_real) + tf.log(epsilon + 1.0 - self.p_fake))
        self.loss_g = -tf.reduce_mean(tf.log(epsilon + self.p_fake))

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        g_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
        d_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')

        optimizer = tf.train.AdamOptimizer()
        with tf.control_dependencies(g_ops):
            self.train_op_g = optimizer.minimize(self.loss_g, var_list=g_vars)
        with tf.control_dependencies(d_ops):
            self.train_op_d = optimizer.minimize(self.loss_d, var_list=d_vars)

        self.summary_op_loss_d = None
        self.summary_op_loss_g = None
        self.summary_op_loss_g_test = None
        self.summarize()

    def summarize(self):
        self.summary_op_loss_d = tf.summary.scalar('loss_D', self.loss_d)
        self.summary_op_loss_g = tf.summary.scalar('loss_G', self.loss_g)
        self.summary_op_loss_g_test = tf.summary.scalar('loss_G_test', self.loss_g)

    def run(self):
        for epoch in range(self.epochs):
            for set_ in [self.data.training, self.data.test]:
                for features, labels, _ in set_:
                    self.batch_counter += 1
                    feed_dict = {self.real: features, self.train_phase: True}
                    _, loss_d_summary = self.session.run([self.train_op_d, self.summary_op_loss_d], feed_dict)
                    self.writer.add_summary(loss_d_summary, self.batch_counter)
                    feed_dict = {self.train_phase: True}
                    _, loss_g_summary, generated_train = self.session.run(
                        [self.train_op_g, self.summary_op_loss_g, self.generated], feed_dict)
                    self.writer.add_summary(loss_g_summary, self.batch_counter)
                    freq = 100
                    if self.batch_counter % freq == 0:
                        self.saver.save(self.session, utils.ckpt())
                        feed_dict = {self.train_phase: False}
                        loss_g_summary_test, generated = self.session.run([self.summary_op_loss_g_test, self.generated],
                                                                          feed_dict)
                        self.writer.add_summary(loss_g_summary_test, self.batch_counter)
                        num = 1
                        for i, img in enumerate(np.concatenate([generated[:num], generated_train[:num], features[:num]],
                                                               axis=2)):
                            save_image(img, self.batch_counter // freq, i)
                    print(self.batch_counter)
                    if self.batch_limit is not None and self.batch_counter >= self.batch_limit:
                        exit()
