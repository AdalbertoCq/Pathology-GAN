import shutil
import argparse
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from data_manipulation.data import Data
from models.generative.gans.BigGAN import BigGAN
from models.generative.utils import *
from models.generative.tools import *
from models.evaluation.features import *

parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
parser.add_argument('--type', dest='type', required= True, help='Type of PathologyGAN: unconditional, er, or survival.')
parser.add_argument('--checkpoint', dest='checkpoint', required= True, help='Path to pre-trained weights (.ckt) of PathologyGAN: unconditional, er, or survival.')
parser.add_argument('--num_samples', dest='num_samples', required= True, help='Number of images to generate.')
parser.add_argument('--batch_size', dest='batch_size', help='Batch size, default size is 64.')
args = parser.parse_args()
pathgan_type = args.type
checkpoint = args.checkpoint
num_samples = int(args.num_samples)
batch_size = args.batch_size

if pathgan_type == 'unconditional':
    conditional = False
    label_dim = None
    label_t = None
elif pathgan_type == 'er':
    conditional = True
    label_dim = 1
    label_t = 'cat'
elif pathgan_type == 'survival':
    conditional = True
    label_dim = 0
    label_t = 'cat'
else:
    print('The available PathologyGAN options are:')
    print('\tunconditional')
    print('\ter')
    print('\tsurvival')
    exit(1)

if batch_size is not None:
    batch_size = int(batch_size)
else:
    batch_size = 50

main_path = os.path.dirname(os.path.realpath(__file__))
dbs_path = os.path.dirname(os.path.realpath(__file__))
    
# Dataset information.
image_width = 224
image_height = 224
image_channels = 3
dataset='vgh_nki'
marker='he'
name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (main_path, name_run)

# Hyperparameters.
batch_size = 64
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

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, empty=True)



with tf.Graph().as_default():
    biggan = BigGAN(data=data, z_dim=z_dim, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2,
                    n_critic=n_critic, gp_coeff=gp_coeff, conditional=conditional, label_dim=label_dim, label_t=label_t, loss_type=loss_type, model_name='PathologyGAN')
    if pathgan_type == 'unconditional':
        gen_hdf5_path = generate_samples_from_checkpoint(biggan, data, data_out_path=main_path, checkpoint=checkpoint, label=label_dim, num_samples=num_samples, batches=batch_size)
        shutil.move('%s/evaluation/%s/vgh_nki/he/h224_w224_n3' % (main_path, biggan.model_name), '%s/evaluation/%s/vgh_nki/he/h224_w224_n3_unconditional' % (main_path, biggan.model_name))

    elif pathgan_type == 'er' or pathgan_type == 'survival':
        # Negative
        gen_hdf5_path = generate_samples_from_checkpoint(biggan, data, data_out_path=main_path, checkpoint=checkpoint, label=0, num_samples=num_samples, batches=batch_size)
        shutil.move('%s/evaluation/%s/vgh_nki/he/h224_w224_n3' % (main_path, biggan.model_name), '%s/evaluation/%s/vgh_nki/he/h224_w224_n3_%s_negative' % (main_path, biggan.model_name, pathgan_type))
        # Positive
        gen_hdf5_path = generate_samples_from_checkpoint(biggan, data, data_out_path=main_path, checkpoint=checkpoint, label=1, num_samples=num_samples, batches=batch_size)
        shutil.move('%s/evaluation/%s/vgh_nki/he/h224_w224_n3' % (main_path, biggan.model_name), '%s/evaluation/%s/vgh_nki/he/h224_w224_n3_%s_positive' % (main_path, biggan.model_name, pathgan_type))


