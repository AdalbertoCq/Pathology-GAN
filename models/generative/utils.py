import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import h5py

# Simple function to plot number images.
def plot_images(plt_num, images, dim, title=None, axis='off'):
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
    plt.plot(losses[:, 0], label='Discriminator', alpha=0.5)
    plt.plot(losses[:, 1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('%s/training_loss.png' % data_out_path)


# Run session to generate output samples.
def show_generated(session, z_input, z_dim, output_fake, n_images, dim=20):
    sample_z = np.random.uniform(low=-1., high=1., size=(n_images, z_dim))
    feed_dict = {z_input:sample_z}
    gen_samples = session.run(output_fake, feed_dict=feed_dict)
    plot_images(plt_num=n_images, images=gen_samples, dim=dim)    
    return gen_samples, sample_z


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
def setup_output(show_epochs, epochs, data, n_images, z_dim, data_out_path, model_name, restore):

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

    size_img = (epochs*data.training.iterations)//show_epochs+1
    img_db_shape = (size_img, n_images, image_height, image_width, image_channels)
    latent_db_shape = (size_img, n_images, z_dim)
    hdf5_gen = h5py.File(gen_images, mode='w')
    hdf5_latent = h5py.File(latent_images, mode='w')
    img_storage = hdf5_gen.create_dataset(name='generated_img', shape=img_db_shape, dtype=np.float32)
    latent_storage = hdf5_latent.create_dataset(name='generated_img', shape=latent_db_shape, dtype=np.float32)

    return img_storage, latent_storage, checkpoints

