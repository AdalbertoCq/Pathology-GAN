import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import manifold

from models.generative.vae_se_densenet_strict.encoder import Encoder
from models.generative.vae_se_densenet_strict.generator import Generator

from preparation.utils import save_image

from models.runner import Runner
import models.utils as utils


class VAE(Runner):
    def __init__(self, bottleneck_dim, enc_details, dec_details, vars_job_id, restore, batch_size, epochs, lr, epsilon=1e-5):
        super().__init__(vars_job_id=vars_job_id, restore=restore, batch_size=batch_size, epochs=epochs, lr=lr)

        self.epsilon = epsilon

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

        '''
        Input image, and Batch Norm trainign flag.
        Shape of the latent space for the given batch size. Default = [50] + (28*28, ) = [50, 784]
        '''
        self.x = tf.placeholder(tf.float32, [None] + self.data.training.shape[1:], name='input_image')
        self.train_phase = tf.placeholder(tf.bool, name='input_bn_boolean')
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
        self.z, self.mean_z_given_xi, self.logs2_z_given_xi = self.sample_z(self.e)
        self.mean_xi_given_z, self.logs2_xi_given_z = vae_generator.build_graph(self.z, self.train_phase)

        '''
        Loss and Optimizer.

        MSE for the image reconstruction.
        KL Divergence between the prior P(Z) and enconding function Q(Z/Xi).
        
        '''
        # KL Divergence.
        z_sigma_2 = tf.exp(self.logs2_z_given_xi)
        z_log_sigma_2 = tf.log(z_sigma_2 + self.epsilon)
        mean_z_2 = tf.square(self.mean_z_given_xi)
        self.kl_divergence = -.5 * tf.reduce_sum(1 + z_log_sigma_2 - mean_z_2 - z_sigma_2, axis=-1)
        self.kl_divergence = tf.reduce_mean(-self.kl_divergence, axis=-1)

        # Recon error from encoding space of datapoint.
        x_sigma_2 = tf.exp(self.logs2_xi_given_z)
        exp_ls2 = tf.square(self.x - self.mean_xi_given_z) / (self.epsilon + x_sigma_2)
        se = tf.log(2 * np.pi) + tf.log(self.epsilon + x_sigma_2) + exp_ls2
        self.sampling_expt = tf.reduce_sum(-.5 * se, axis=[1, 2, 3])
        self.sampling_expt = tf.reduce_mean(self.sampling_expt, axis=-1)

        elbo = self.kl_divergence + self.sampling_expt
        self.vae_loss = -elbo

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = optimizer.minimize(self.vae_loss)

        # Latent input and image generation.
        self.latent = tf.placeholder(dtype=tf.float32, shape=self.shape, name='input_latent_vector')
        self.gen, _ = vae_generator.build_graph(self.latent, self.train_phase, reuse=True)

        self.summary_op = None
        self.summary_images = None
        self.summary_gen = None
        self.summarize()

    # Parametrization trick, sample from normal and scale.
    def sample_z(self, encoding):
        mean_z_given_xi = encoding[:, :, 0]
        logs2_z_given_xi = encoding[:, :, 1]
        sigma_z_given_xi = tf.sqrt(tf.exp(logs2_z_given_xi))
        z = mean_z_given_xi + sigma_z_given_xi * self.rand
        return z, mean_z_given_xi, sigma_z_given_xi

    # Summarize losses for tensorboard.
    def summarize(self):
        # Scalars
        tf.summary.scalar('Loss/Reconstruction', self.sampling_expt)
        tf.summary.scalar('Loss/KL_Divergence', self.kl_divergence)
        tf.summary.scalar('Loss/VAE_loss', self.vae_loss)

        # Histograms
        tf.summary.histogram("Encoder/Mean", self.mean_z_given_xi)
        tf.summary.histogram("Encoder/LogSimaSq", self.logs2_z_given_xi)
        tf.summary.histogram("Z_sample", self.z)
        tf.summary.histogram("Generator/Mean", self.mean_xi_given_z)
        tf.summary.histogram("Generator/LogSimaSq", self.logs2_xi_given_z)
        tf.summary.histogram("Generator/Original_Image", self.x)

        # Summary.
        self.summary_op = tf.summary.merge_all()

        # Images
        self.summary_orecon = tf.summary.image('Recon/Input_image', self.x)
        self.summary_irecon = tf.summary.image('Recon/Output_image', self.mean_xi_given_z)
        self.summary_images = tf.summary.merge([self.summary_orecon, self.summary_irecon])


        self.summary_gen = tf.summary.image("Gen/Output_image", self.gen)

    # Training VAE.
    def run(self):  # run_train
        for epoch in range(self.epochs):
            # TODO: Is it running training for both? 
            # Might be good to have some test data and see how it construct it.
            for set_ in [self.data.training, self.data.test]:
                for features, labels, _ in set_:
                    self.batch_counter += 1
                    freq_img = 100
                    freq_tsne = 100

                    # TF Train and add to summary.
                    _, summary = self.session.run([self.train_op, self.summary_op], feed_dict={self.x: features, self.train_phase: True})
                    self.writer.add_summary(summary, self.batch_counter)
                    
                    # Dump out images to track progress.
                    if self.batch_counter % freq_img == 0:
                        # Reconstruct image.
                        summary, summary_images, reconstructed = self.session.run([self.summary_op, self.summary_images, self.mean_xi_given_z],
                                                                  feed_dict={self.x: features, self.train_phase: False})
                        self.writer.add_summary(summary, self.batch_counter)
                        self.writer.add_summary(summary_images, self.batch_counter)

                        num = 1
                        for i, pair in enumerate(zip(features[:num], reconstructed[:num])):
                            concatenated = np.concatenate(pair, axis=1)
                            save_image(concatenated, self.vars_job_id, '%d_%d' % (self.batch_counter // freq_img, i))

                        # Generate images.
                        summary_gen = self.gen_samples('gen_%d_%d' % (self.batch_counter // freq_img, i))
                        self.writer.add_summary(summary_gen, self.batch_counter)
                        self.saver.save(self.session, utils.ckpt(self.vars_job_id))

                    # Run T-SNE.
                    if self.batch_counter % freq_tsne == 0:
                        utils.run_tsne(self, 500, perplexity=30, learning_step=10, iterations=10000, name='tsne_%d_%d_p%s' % (self.batch_counter // freq_tsne, i, 30))
                        utils.run_tsne(self, 500, perplexity=50, learning_step=10, iterations=10000, name='tsne_%d_%d_p%s' % (self.batch_counter // freq_tsne, i, 50))
                    print(self.batch_counter)


    # Auxiliary function to plot generated digits.
    def gen_samples(self, name, num_samples=25):
        # Normal distribution to generate images.
        latent = np.random.normal(loc=0.0, scale=1.0, size=self.shape)
        summary_gen, generated = self.session.run([self.summary_gen, self.gen], feed_dict={self.train_phase: False, self.latent: latent})

        if int(generated.shape[0]) > num_samples:
            generated = generated[:num_samples, :, :, :]

        n_sqrt = int(np.sqrt(generated.shape[0]))
        fig, axes = plt.subplots(n_sqrt, n_sqrt, sharex=True, sharey=True, figsize=(n_sqrt * 3, n_sqrt * 3))
        for ii, ax in zip(range(0, generated.shape[0]), axes.flatten()):
            ax.imshow(generated[ii, :, :], aspect='equal')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig('run/%s/img/%s.png' % (self.vars_job_id, name))
        plt.close()
        return summary_gen

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
                reconstructed = self.session.run(self.mean_xi_given_z, feed_dict)[0]

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
