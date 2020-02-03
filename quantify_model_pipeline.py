import tensorflow as tf
import os 
import argparse
from data_manipulation.data import Data
from models.generative.gans.BigGAN import BigGAN
from models.generative.gans.PathologyGAN import PathologyGAN
from models.evaluation.features import *
from models.score.score import Scores
from models.score.crimage_score import CRImage_Scores


parser = argparse.ArgumentParser(description='StylePathologyGAN fake image generator and feature extraction.')
parser.add_argument('--checkpoint', dest='checkpoint', required= True, help='Path to pre-trained weights (.ckt) of PathologyGAN.')
parser.add_argument('--num_samples', dest='num_samples', required= True, type=int, help='Number of images to generate.')
parser.add_argument('--batch_size', dest='batch_size', required= True, type=int, help='Batch size.')
parser.add_argument('--z_dim', dest='z_dim', required= True, type=int, help='Latent space size.')
# parser.add_argument('--create_samples', dest='create_samples', action='store_true', help='Boolean flag to control sample creation.')
args = parser.parse_args()
checkpoint = args.checkpoint
num_samples = args.num_samples
batch_size = args.batch_size
z_dim = args.z_dim
# create_samples = args.create_samples

model = 'StylePathologyGAN'

main_path = os.path.dirname(os.path.realpath(__file__))
dbs_path = os.path.dirname(os.path.realpath(__file__))

# Dataset information.
crimage_path = os.path.join(main_path, 'CRImage')
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
image_width = 224
image_height = 224
image_channels = 3
dataset='vgh_nki'
marker='he'
offset = None
if offset is not None:
	name_run = 'h%s_w%s_n%s_%s' % (image_height, image_width, image_channels, offset)
else:
	name_run = 'h%s_w%s_n%s' % (image_height, image_width, image_channels)
data_out_path = '%s/%s' % (data_out_path, name_run)

# Hyperparameters.
learning_rate_g = 1e-4
learning_rate_d = 1e-4
learning_rate_e = 1e-4
beta_1 = 0.5
beta_2 = 0.9
restore = False

# Model
layers = 5
# z_dim = 200
alpha = 0.2
n_critic = 5
gp_coeff = .65
use_bn = False
loss_type = 'relativistic gradient penalty'


data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path)

with tf.Graph().as_default():
    if 'StylePathologyGAN' in checkpoint:
    	# Instantiate StylePathologyGAN
        pathgan = PathologyGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, loss_type=loss_type, model_name=model)

    elif 'PathologyGAN' in checkpoint:
        # Instantiate PathologyGAN
        pathgan = BigGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, beta_2=beta_2, n_critic=n_critic, gp_coeff=gp_coeff, conditional=False, label_dim=None, label_t=None, loss_type=loss_type, model_name='PathologyGAN')

    # Generate Fake samples from checkpoint.
    gen_hdf5_path = generate_samples_from_checkpoint(model=pathgan, data=data, data_out_path=main_path, checkpoint=checkpoint, num_samples=num_samples, batches=batch_size)

# Generate Inception features from fake images.
with tf.Graph().as_default():
    hdf5s_features = inception_tf_feature_activations(hdf5s=[gen_hdf5_path], input_shape=data.training.shape[1:], batch_size=50)


# Generate CRImage features from fake images. 
main_evaluation_path = gen_hdf5_path.split('hdf5_')[0]
crimage_file = os.path.join(main_evaluation_path, 'crimage_train.txt')
if not os.path.isfile(crimage_file):
	crimage_command = 'cd %s; Rscript --vanilla crimage_style.r %s %s' % (crimage_path, main_evaluation_path, num_samples)
	os.system(crimage_command)

nki_vgh_new_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3//hdf5_nki_vgh_he_features_train_real.h5'
nki_vgh_new_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/hdf5_nki_vgh_he_features_test_real.h5'

nki_vgh_new_train_cr = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3//crimage_train.txt'
nki_vgh_new_test_cr = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/crimage_test.txt'

print('----- INCEPTION FEATURES -----')
print('Study on VGH_NKI')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_train, nki_vgh_new_test, 'Real Train VGH_NKI', 'Real Test VGH_NKI', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_train, hdf5s_features[0], 'Real Train', pathgan.model_name, k=1, display=True)
    scores.run_scores()

with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_test, hdf5s_features[0], 'Real Test', pathgan.model_name, k=1, display=True)
    scores.run_scores()

print('----- CRIMAGE FEATURES -----')
with tf.Graph().as_default():
    scores = CRImage_Scores(ref1_crimage=nki_vgh_new_train_cr, ref2_crimage=crimage_file, name_x='Real Train', name_y=pathgan.model_name, k=1, GPU=True, display=True)
    scores.run_crimage_scores()

with tf.Graph().as_default():
    scores = CRImage_Scores(ref1_crimage=nki_vgh_new_test_cr, ref2_crimage=crimage_file, name_x='Real Test', name_y=pathgan.model_name, k=1, GPU=True, display=True)
    scores.run_crimage_scores()
