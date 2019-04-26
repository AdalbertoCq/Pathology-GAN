import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from models.generative.utils import *
from data_manipulation.data import Data
from models.generative.tools import *
from data_manipulation.utils import write_sprite_image
from tensorflow.contrib.tensorboard.plugins import projector
from models.generative.utils import *
import platform
from models.generative.gans.InfoBigGAN import InfoBigGAN
from models.generative.gans.BigGAN import BigGAN
from models.generative.gans.SAGAN import SAGAN

model = 'SAGAN'
model = 'BigGAN'

if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/' + model
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/data_model_output/' + model

# Dataset information.
image_width = 224
image_height = 224
image_channels = 3
dataset='vgh_nki'
marker='he'
name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (main_path, name_run)

# Hyperparameters.
epochs = 15
batch_size = 64
learning_rate_g = 1e-4
learning_rate_d = 1e-4
beta_1 = 0.5
beta_2 = 0.9
restore = True

# Model
layers = 5
z_dim = 128
alpha = 0.2
n_critic = 5
gp_coeff = .5
use_bn = False
loss_type = 'relativistic gradient penalty'
# loss_type = 'relativistic standard'
# loss_type = 'hinge'

# Conditional
conditional = True
label_dim = 1
label_t = 'cat'

# InfoGAN
delta = 1.0
c_dim = 10


data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size)

with tf.Graph().as_default():
    # biggan = InfoBigGAN(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2,
    #               		c_dim=c_dim, delta=delta, n_critic=n_critic, gp_coeff=gp_coeff, conditional=conditional, label_dim=label_dim, loss_type=loss_type)

    biggan = BigGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2,
                  	n_critic=n_critic, gp_coeff=gp_coeff, conditional=conditional, label_dim=label_dim, label_t=label_t, loss_type=loss_type)

    losses = biggan.train(epochs, data_out_path, data, restore, n_images=10, show_epochs=None)
