from data_manipulation.data import Data
from models.evaluation.features import *
import argparse

parser = argparse.ArgumentParser(description='PathologyGAN fake image generator.')
parser.add_argument('--num_samples', dest='num_samples', required=False, type=int, default=5000, help='Number of images to generate.')
parser.add_argument('--batch_size', dest='batch_size', required=False, type=int, default=50, help='Batch size.')
parser.add_argument('--main_path', dest='main_path', default=None, type=str, help='Path for the output run.')
parser.add_argument('--dbs_path', dest='dbs_path', type=str, default=None, help='Directory with DBs to use.')
parser.add_argument('--img_size', dest='img_size', type=int, default=224, help='Image size for the model.')
parser.add_argument('--dataset', dest='dataset', type=str, default='vgh_nki', help='Dataset to use.')
parser.add_argument('--marker', dest='marker', type=str, default='he', help='Marker of dataset to use.')
args = parser.parse_args()
num_samples = args.num_samples
batch_size = args.batch_size
dataset = args.dataset
marker = args.marker
img_size = args.img_size
main_path = args.main_path
dbs_path = args.dbs_path

if main_path is None:
	main_path = os.path.dirname(os.path.realpath(__file__))
if dbs_path is None:
	dbs_path = os.path.dirname(os.path.realpath(__file__))

image_width = img_size
image_height = img_size
image_channels = 3

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path, labels=False)
hdf5s = real_samples(data=data, data_output_path=main_path, num_samples=num_samples)

with tf.Graph().as_default():
    inception_tf_feature_activations(hdf5s=hdf5s, input_shape=data.training.shape[1:], batch_size=batch_size)