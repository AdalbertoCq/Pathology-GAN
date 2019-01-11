import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import manifold

from models.generative.vae_se_densenet.encoder import Encoder
from models.generative.vae_se_densenet.generator import Generator

from preparation.utils import save_image

from models.runner import Runner
import models.utils as utils


class VAE(Runner):
    def __init__(self, bottleneck_dim, enc_details, dec_details, kl_loss_factor, vars_job_id, restore, batch_size, epochs, lr):
        super().__init__(vars_job_id=vars_job_id, restore=restore, batch_size=batch_size, epochs=epochs, lr=lr)

        # Parameters for Encoder.
        self.enc_blocks = enc_details['enc_blocks']
        self.enc_growth_rates = enc_details['enc_growth_rates']
        self.enc_final_filters = enc_details['enc_final_filters']
        self.enc_shape = enc_details['enc_shape']
        self.enc_downsample = enc_details['downsample']
        self.enc_activation = enc_details['activation']

        # Parameters for Generator.
        self.gen_blocks = dec_details['gen_blocks']
        self.gen_growth_rates = dec_details['gen_growth_rates']
        self.gen_final_filters = dec_details['gen_final_filters']
        self.gen_shape = dec_details['gen_shape']
        self.gen_downsample = dec_details['downsample']

        # Encoding dimension 56*56=3136
        self.bottleneck = (bottleneck_dim * bottleneck_dim,)

        # Factor to weight the KL Divergence between the prior P(Z) and enconding function Q(Z/Xi).
        self.kl_loss_factor = kl_loss_factor

        '''
        Input image, and Batch Norm trainign flag.
        Shape of the latent space for the given batch size. Default = [50] + (28*28, ) = [50, 784]
        '''
        self.x = tf.placeholder(tf.float32, [None] + self.data.training.shape[1:])
        self.train_phase = tf.placeholder(tf.bool)
        self.shape = [self.batch_size] + list(self.bottleneck)

        '''
        Building network.
        
        Encoder Q(Z/Xi) --> parametrization_trick --> Generator P(Xi/Z).
        '''
        vae_encoder = Encoder(self.enc_blocks, self.enc_growth_rates, self.enc_final_filters, self.enc_shape,
                              self.enc_downsample, self.enc_activation)
        vae_generator = Generator(self.gen_blocks, self.gen_growth_rates, self.gen_final_filters, self.gen_shape,
                                  self.gen_downsample)

        self.e = vae_encoder.build_graph(self.x, self.train_phase)
        self.rand = tf.random_normal(self.shape)
        self.z, self.mean, self.log_std = self.sample_z(self.e)
        self.y = vae_generator.build_graph(self.z, self.train_phase)

        '''
        Loss and Optimizer.

        MSE for the image reconstruction.
        KL Divergence between the prior P(Z) and enconding function Q(Z/Xi).
        
        '''
        self.img_loss = tf.reduce_mean((self.y - self.x)**2)
        self.kl_loss = tf.reduce_mean((tf.exp(self.log_std)**2 + self.mean**2 - 2 * self.log_std - 1) / 2)
        self.loss = self.img_loss + self.kl_loss_factor * self.kl_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = optimizer.minimize(self.loss)

        # Latent input and image generation.
        self.latent = tf.placeholder(dtype=tf.float32, shape=self.shape)
        self.gen = vae_generator.build_graph(self.latent, self.train_phase, reuse=True)

        self.summary_op = None
        self.test_summary_op = None
        self.summarize()

    # Parametrization trick, sample from normal and scale.
    def sample_z(self, encoding):
        mean = encoding[:, :, 0]
        log_std = encoding[:, :, 1]
        std = tf.exp(log_std)
        z = mean + std * self.rand
        return z, mean, log_std

    # Summarize losses for tensorboard.
    def summarize(self):
        tf.summary.scalar('reconstruction', self.img_loss)
        tf.summary.scalar('KL-divergence', self.kl_loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.image('Input_image', self.x)
        tf.summary.image('Output_image', self.y)
        self.summary_op = tf.summary.merge_all()
        self.test_summary_op = tf.summary.scalar('loss_test', self.loss)

    # Training VAE.
    def run(self):  # run_train
        for epoch in range(self.epochs):
            # TODO: Is it running training for both? 
            # Might be good to have some test data and see how it construct it.
            for set_ in [self.data.training, self.data.test]:
                for features, labels, _ in set_:
                    self.batch_counter += 1
                    freq_img = 100
                    freq_tsne = 1000

                    # TF Train and add to summary.
                    _, summary = self.session.run([self.train_op, self.summary_op], feed_dict={self.x: features, self.train_phase: True})
                    self.writer.add_summary(summary, self.batch_counter)
                    
                    # Dump out images to track progress.
                    if self.batch_counter % freq_img == 0:
                        # Reconstruct image.
                        test_summary, generated = self.session.run([self.test_summary_op, self.y], feed_dict={self.x: features, self.train_phase: False})
                        self.writer.add_summary(test_summary, self.batch_counter)
                        self.saver.save(self.session, utils.ckpt(self.vars_job_id))
                        num = 1
                        for i, pair in enumerate(zip(features[:num], generated[:num])):
                            concatenated = np.concatenate(pair, axis=1)
                            save_image(concatenated, self.vars_job_id, '%d_%d' % (self.batch_counter // freq_img, i))
                        # Generate images.
                        utils.gen_samples(self, 'gen_%d_%d' % (self.batch_counter // freq_img, i))

                    # Run T-SNE.
                    if self.batch_counter % freq_tsne == 0:
                        utils.run_tsne(self, 500, perplexity=30, learning_step=10, iterations=10000, name='tsne_%d_%d_p%s' % (self.batch_counter // freq_tsne, i, 30))
                        utils.run_tsne(self, 500, perplexity=50, learning_step=10, iterations=10000, name='tsne_%d_%d_p%s' % (self.batch_counter // freq_tsne, i, 50))
                    print(self.batch_counter)

    # Generate images: Num of images=batches.
    def run_gen(self):  # run_gen
        num = 1
        for epoch in range(self.epochs):
            for features, labels, _ in self.data.training:
                self.batch_counter += 1
                latent = np.random.normal(loc=0.0, scale=1.0, size=self.shape)
                generated = self.session.run([self.gen], feed_dict= {self.train_phase: False, self.latent: latent})
                latent = np.reshape(latent, enc_shape)
                for i, (n, g) in enumerate(zip(latent[:num], generated[:num])):
                    save_image(g[i, :, :, :], self.vars_job_id, 'gen_%d_%d' % (self.batch_counter, i), train=False)
                print(self.batch_counter)

    # T-SNE run.
    def run_tsne(self):
        utils.run_tsne(self, 1000, perplexity=40, learning_step=10, iterations=10000, name='tsne_standalone_p%s' % 40)

    # PCA run.
    def run_trans(self):  # run_trans
        for epoch in range(self.epochs):
            enc_arr = []
            feature_arr = []
            # Gets encodings for 30 batches.
            # TODO: Enough batches?
            for features, labels, _ in self.data.training:
                self.batch_counter += 1
                feed_dict = {self.x: features, self.train_phase: False}
                encodings = self.session.run(self.z, feed_dict)
                enc_arr.append(encodings)
                feature_arr.append(features)
                print(self.batch_counter)
                if self.batch_counter == 30:
                    break

            # Runs PCA on encodings.
            encodings = np.reshape(np.array(enc_arr), [-1, 56 * 56])
            features = np.reshape(np.array(feature_arr), [-1, 224, 224, 3])
            points = decomposition.PCA(n_components=1).fit_transform(encodings)
            lower = np.argmin(points)
            upper = np.argmax(points)
            _, (a, b) = plt.subplots(2, 1)
           

            # TODO: Is he trying to compare this two?
            # TODO: Checking delta between the max-min. Purpose?
            n = 8
            first = encodings[None, lower]
            last = encodings[None, upper]
            delta = last - first            
            ys = []
            for i in range(n):
                latent = first + i * delta / (n - 1)
                latent = np.tile(latent, [self.batch_size, 1])
                feed_dict = {self.latent: latent, self.train_phase: False}
                y = self.session.run(self.gen, feed_dict)
                ys.append(y[0])
            img = np.concatenate(ys, axis=1)
            a.imshow(img)

            # TODO: Checking delta between the max-min. Purpose?
            first = features[lower]
            last = features[upper]
            delta = last - first
            imgs = []
            for i in range(n):
                img = first + i * delta / (n - 1)
                imgs.append(img)
            img = np.concatenate(imgs, axis=1)
            b.imshow(img)
            plt.show()


    # 
    def run_vec_alg(self):  # run_vec_alg
        for epoch in range(self.epochs):
            for features, labels, _ in self.data.training:
                self.batch_counter += 1

                # Get encoding for each training sample.
                feed_dict = {self.x: features, self.train_phase: False}
                encoding = self.session.run(self.z, feed_dict)

                # Gets 3 samples
                a, b, c = encoding[0:1], encoding[1:2], encoding[2:3]

                # Linear combination of these?
                combinations = np.tile(a - b + c, [self.batch_size, 1])
                print(combinations.shape, 'shape', self.batch_size, a.shape, (a - b + c).shape)

                # Reconstruction of the linear combination, relationship between images?
                feed_dict = {self.latent: combinations, self.train_phase: False}
                reconstructed = self.session.run(self.y, feed_dict)[0]

                # Plots.
                _, subplots = plt.subplots(1, 3)
                img_titles = ['A', 'B', 'C']
                for subplot, img, title in zip(subplots, features[:3], img_titles):
                    subplot.imshow(img)
                    subplot.set_title(title)
                    subplot.set_xticks([])
                    subplot.set_yticks([])
                plt.subplots_adjust(hspace=0.01, wspace=0.01)
                plt.show()
                _, (a, b) = plt.subplots(1, 2)
                a.imshow(reconstructed)
                a.set_title('A - B + C (latent)')
                a.set_xticks([])
                a.set_yticks([])
                b.imshow(features[0] - features[1] + features[2])
                b.set_title('A - B + C (real)')
                b.set_xticks([])
                b.set_yticks([])
                plt.subplots_adjust(hspace=0.01, wspace=0.1)
                plt.show()
                print(self.batch_counter)
