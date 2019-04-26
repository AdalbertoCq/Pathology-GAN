import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from models.generative.utils import *
from data_manipulation.data import Data
from models.generative.tools import *
from models.generative.utils import *
import os
import platform
from models.generative.gans.BigGAN import BigGAN
from models.evaluation.features import *

if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative/'
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/'
    
# Dataset information.
image_width = 224
image_height = 224
image_channels = 3
dataset='vgh_nki'
marker='he'
name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (main_path, name_run)

# Hyperparameters.
batch_size = 50
learning_rate_g = 1e-4
learning_rate_d = 1e-4
beta_1 = 0.5
beta_2 = 0.9

# Model
z_dim = 128
alpha = 0.2
n_critic = 5
gp_coeff = .5
use_bn = False
loss_type = 'relativistic gradient penalty'

# Conditional
conditional = False
label_dim = 1
label_t = 'cat'

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=main_path)
hdf5_images_train_real, hdf5_images_test_real = real_samples(data=data, data_output_path=main_path)
with tf.Graph().as_default():
    biggan = BigGAN(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, conditional=conditional, label_dim=label_dim, label_t=label_t, loss_type=loss_type)
    hdf5_images_generated = generate_samples(biggan, data, data_out_path=main_path, num_samples=5000, batches=50)

hdf5s = [hdf5_images_generated, hdf5_images_train_real, hdf5_images_test_real]
with tf.Graph().as_default():
    inception_tf_feature_activations(hdf5s=hdf5s, input_shape=data.training.shape[1:], batch_size=50, checkpoint_path=None)
    # inception_feature_activations(hdf5s=hdf5s, input_shape=data.training.shape[1:], batch_size=50, checkpoint_path=None)

