import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os 
# from models.generative.utils import *
from data_manipulation.data import Data
# from models.generative.tools import *
# from data_manipulation.utils import write_sprite_image
# from tensorflow.contrib.tensorboard.plugins import projector
# from models.generative.utils import *
# import platform
from models.generative.gans.BigGAN import BigGAN
import argparse

parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
parser.add_argument('--type', dest='type', required= True, help='Type of PathologyGAN: unconditional, er, or survival.')
parser.add_argument('--epochs', dest='epochs', help='Number epochs to run: default is 45 epochs.')
parser.add_argument('--batch_size', dest='batch_size', help='Batch size, default size is 64.')
args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
pathgan_type = args.type

if pathgan_type == 'unconditional':
	conditional = False
	label_dim = None
	label_t = None
elif pathgan_type == 'er':
	conditional = True
	label_dim = 1
	label_t = 'mlp'
elif pathgan_type == 'survival':
	conditional = True
	label_dim = 0
	label_t = 'mlp'
else:
	print('The available PathologyGAN options are:')
	print('\tunconditional')
	print('\ter')
	print('\tsurvival')
	exit(1)

if epochs is not None:
	epochs = int(epochs)
else:
	epochs = 45

if batch_size is not None:
	batch_size = int(batch_size)
else:
	batch_size = 64


model = 'BigGAN'

main_path = os.path.dirname(os.path.realpath(__file__))
dbs_path = os.path.dirname(os.path.realpath(__file__))

# Dataset information.
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
image_width = 224
image_height = 224
image_channels = 3
dataset='vgh_nki'
marker='he'
name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (data_out_path, name_run)

# Hyperparameters.
learning_rate_g = 1e-4
learning_rate_d = 1e-4
beta_1 = 0.5
beta_2 = 0.9
restore = False

# Model
layers = 5
z_dim = 128
alpha = 0.2
n_critic = 5
gp_coeff = .5
use_bn = False
loss_type = 'relativistic gradient penalty'

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path)

with tf.Graph().as_default():
    biggan = BigGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2,
                  	n_critic=n_critic, gp_coeff=gp_coeff, conditional=conditional, label_dim=label_dim, label_t=label_t, loss_type=loss_type, model_name='PathologyGAN')

    losses = biggan.train(epochs, data_out_path, data, restore, n_images=10, show_epochs=None)
