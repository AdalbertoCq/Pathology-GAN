import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import shutil
import h5py
import tensorflow as tf


# Simple function to plot number images.
def plot_images(plt_num, images, dim, title=None, axis='off', row_n=None):
    # Standard parameters for the plot.
    
    mpl.rcParams['figure.figsize'] = dim, dim
    mpl.rcParams['axes.titlepad'] = 20 
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)

    for i in range(0, plt_num):
        fig.add_subplot(1, 10, i+1)
        img = images[i, :, :, :]
        plt.imshow(img)
        plt.axis(axis)
    plt.show()


# Plot and save figure of losses.
def save_loss(losses, data_out_path, dim):    
    mpl.rcParams["figure.figsize"] = dim, dim
    plt.rcParams.update({'font.size': 22})
    losses = np.array(losses)
    num_loss = losses.shape[1]
    for _ in range(num_loss):
        if _ == 0:
            label = 'Generator'
        elif _ == 1:
            label = 'Discriminator'
        else:
            label = 'Mutual Information'
        plt.plot(losses[:, _], label=label, alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('%s/training_loss.png' % data_out_path)


def get_checkpoint(data_out_path, which=0):
    checkpoints_path = os.path.join(data_out_path, 'checkpoints')
    checkpoints = os.path.join(checkpoints_path, 'checkpoint')
    index = 0
    with open(checkpoints, 'r') as f:
        for line in reversed(f.readlines()):
            if index == which:
                return line.split('"')[1]
    print('No model to restore')
    exit()


# Method to setup 
def setup_output(show_epochs, epochs, data, n_images, z_dim, data_out_path, model_name, restore, save_img):

    checkpoints_path = os.path.join(data_out_path, 'checkpoints')
    checkpoints = os.path.join(checkpoints_path, '%s.ckt' % model_name)
    gen_images_path = os.path.join(data_out_path, 'images')
    gen_images = os.path.join(gen_images_path, 'gen_images.h5')
    latent_images = os.path.join(gen_images_path, 'latent_images.h5')
    
    if os.path.isdir(gen_images_path):
         shutil.rmtree(gen_images_path)
    os.makedirs(gen_images_path)
    
    if not restore:
        if os.path.isdir(checkpoints_path):
            shutil.rmtree(checkpoints_path)
        os.makedirs(checkpoints_path)
    

    image_height = data.training.patch_h
    image_width = data.training.patch_w
    image_channels = data.training.n_channels

    if save_img:
        size_img = (epochs*data.training.iterations)//show_epochs+1
        img_db_shape = (size_img, n_images, image_height, image_width, image_channels)
        latent_db_shape = (size_img, n_images, z_dim)
        hdf5_gen = h5py.File(gen_images, mode='w')
        hdf5_latent = h5py.File(latent_images, mode='w')
        img_storage = hdf5_gen.create_dataset(name='generated_img', shape=img_db_shape, dtype=np.float32)
        latent_storage = hdf5_latent.create_dataset(name='generated_img', shape=latent_db_shape, dtype=np.float32)
    else: 
        img_storage = None
        latent_storage = None

    return img_storage, latent_storage, checkpoints


# Run session to generate output samples.
def show_generated(session, z_input, z_dim, output_fake, n_images, c_input=None, c_dim=None, dim=20, show=True):
    gen_samples = list()
    sample_z = list()
    batch_sample = 20
    for x in range(n_images):
        rand_sample = random.randint(0,batch_sample-1)
        
        z_batch = np.random.uniform(low=-1., high=1., size=(batch_sample, z_dim))
        feed_dict = {z_input:z_batch}
        if c_input is not None:
            c_batch = np.random.normal(loc=0.0, scale=1.0, size=(batch_sample, c_dim))
            feed_dict[c_input] = c_batch
        gen_batch = session.run(output_fake, feed_dict=feed_dict)
        gen_samples.append(gen_batch[rand_sample, :, :, :])
        sample_z.append(z_batch[rand_sample, :])
    if show:
        plot_images(plt_num=n_images, images=np.array(gen_samples), dim=dim)    
    return np.array(gen_samples), np.array(sample_z)


# Method to report parameter in the run.
def report_parameters(model, epochs, restore, data_out_path):
    with open('%s/run_parameters.txt' % data_out_path, 'w') as f:
        f.write('Epochs: %s\n' % (epochs))
        f.write('Restore: %s\n' % (restore))
        for attr, value in model.__dict__.items():
            f.write('%s: %s\n' % (attr, value))


