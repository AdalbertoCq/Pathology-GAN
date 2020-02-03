from data_manipulation.data import Data
from models.evaluation.features import *

data_path = os.path.dirname(os.path.realpath(__file__))

image_width = 224
image_height = 224
image_channels = 3
dataset='vgh_nki'
marker='he'

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=data_path, labels=False)
hdf5_images_train_real, hdf5_images_test_real = real_samples(data=data, data_output_path=data_path)
hdf5s = [hdf5_images_train_real, hdf5_images_test_real]

with tf.Graph().as_default():
    inception_tf_feature_activations(hdf5s=hdf5s, input_shape=data.training.shape[1:], batch_size=50, checkpoint_path=None)

