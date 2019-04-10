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


from models.generative.gans.InfoSAGAN import InfoSAGAN
model = 'InfoSAGAN'

from models.generative.gans.BigGAN import BigGAN
model = 'BigGAN'


if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/' + model
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/data_model_output/' + model
    
# Dataset information.
image_width = 224
image_height = 224
image_channels = 3
dataset='nki'
marker='he'
name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (main_path, name_run)

# Hyperparameters.
epochs = 43
batch_size = 64
z_dim = 100
c_dim = 20
conditional = True
label_dim = 5
learning_rate_g = 5e-5
learning_rate_d = 1e-4
alpha = 0.2
beta_1 = 0.5
beta_2 = 0.9
n_critic = 5
gp_coeff = .5
delta = 2.
use_bn = False
loss_type = 'hinge'


restore = True

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size)

# with tf.device("/gpu:0"):
with tf.Graph().as_default():
    # infosagan = InfoSAGAN(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, power_iterations=1, 
    #               		  c_dim=c_dim, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, delta=delta, loss_type='relativistic gradient penalty infogan')
    # losses = infosagan.train(epochs, data_out_path, data, restore, n_images=10, show_epochs=None)

    biggan = BigGAN(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2,
                  n_critic=n_critic, gp_coeff=gp_coeff, conditional=conditional, label_dim=label_dim, loss_type=loss_type)

    losses = biggan.train(epochs, data_out_path, data, restore, n_images=10, show_epochs=None)
