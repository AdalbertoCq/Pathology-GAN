import tensorflow as tf
import os 
import argparse
from data_manipulation.data import Data
from models.generative.gans.PathologyGAN import PathologyGAN
from models.evaluation.features import *


parser = argparse.ArgumentParser(description='PathologyGAN fake image generator.')
parser.add_argument('--checkpoint', dest='checkpoint', required=True, help='Path to pre-trained weights (.ckt) of PathologyGAN.')
parser.add_argument('--num_samples', dest='num_samples', required=False, type=int, default=5000, help='Number of images to generate.')
parser.add_argument('--batch_size', dest='batch_size', required=False, type=int, default=50, help='Batch size.')
parser.add_argument('--z_dim', dest='z_dim', required=True, type=int, default=200, help='Latent space size.')
parser.add_argument('--dataset', dest='dataset', type=str, default='vgh_nki', help='Dataset to use.')
parser.add_argument('--marker', dest='marker', type=str, default='he', help='Marker of dataset to use.')
parser.add_argument('--img_size', dest='img_size', type=int, default=224, help='Image size for the model.')
parser.add_argument('--main_path', dest='main_path', default=None, type=str, help='Path for the output run.')
parser.add_argument('--dbs_path', dest='dbs_path', type=str, default=None, help='Directory with DBs to use.')
parser.add_argument('--model', dest='model', type=str, default='PathologyGAN', help='Model name.')
args = parser.parse_args()
checkpoint = args.checkpoint
num_samples = args.num_samples
batch_size = args.batch_size
z_dim = args.z_dim
dataset = args.dataset
marker = args.marker
img_size = args.img_size
main_path = args.main_path
dbs_path = args.dbs_path
model = args.model

if main_path is None:
	main_path = os.path.dirname(os.path.realpath(__file__))
if dbs_path is None:
	dbs_path = os.path.dirname(os.path.realpath(__file__))

# Dataset information.
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
image_width = img_size
image_height = img_size
image_channels = 3
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
    pathgan = PathologyGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, loss_type=loss_type, model_name='PathologyGAN')
    gen_hdf5_path = generate_samples_from_checkpoint(model=pathgan, data=data, data_out_path=main_path, checkpoint=checkpoint, num_samples=num_samples, batches=batch_size)

# Generate Inception features from fake images.
with tf.Graph().as_default():
    hdf5s_features = inception_tf_feature_activations(hdf5s=[gen_hdf5_path], input_shape=data.training.shape[1:], batch_size=batch_size)
