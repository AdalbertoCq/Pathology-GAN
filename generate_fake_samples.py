import tensorflow as tf
import os 
import argparse
from data_manipulation.data import Data
from models.generative.gans.PathologyGAN import PathologyGAN
from models.evaluation.features import *


parser = argparse.ArgumentParser(description='PathologyGAN fake image generator.')
parser.add_argument('--checkpoint', dest='checkpoint', required= True, help='Path to pre-trained weights (.ckt) of PathologyGAN.')
parser.add_argument('--num_samples', dest='num_samples', required= True, type=int, default=5000, help='Number of images to generate.')
parser.add_argument('--batch_size', dest='batch_size', required= True, type=int, default=50, help='Batch size.')
parser.add_argument('--z_dim', dest='z_dim', required= True, type=int, default=200, help='Latent space size.')
parser.add_argument('--model', dest='model', type=str, default='PathologyGAN', help='Model name.')
args = parser.parse_args()
checkpoint = args.checkpoint
num_samples = args.num_samples
batch_size = args.batch_size
z_dim = args.z_dim
model = args.model

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
offset = None
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
alpha = 0.2
n_critic = 5
gp_coeff = .65
use_bn = False
loss_type = 'relativistic gradient penalty'

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path)

with tf.Graph().as_default():
    pathgan = PathologyGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, loss_type=loss_type, model_name=model)
