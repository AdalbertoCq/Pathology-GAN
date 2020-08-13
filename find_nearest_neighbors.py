from models.evaluation.tools import *
import argparse

parser = argparse.ArgumentParser(description='PathologyGAN nearest neighbors finder.')
parser.add_argument('--out_path', dest='out_path', required=True, help='Output path folder for images.')
parser.add_argument('--real_features', dest='real_features_hdf5', required=True, help='Path to the real features HDF5 file.')
parser.add_argument('--fake_features', dest='gen_features_hdf5', required=True, help='Path to the fake features HDF5 file.')
parser.add_argument('--real_images', dest='real_images_hdf5', required=True, help='Path to the real images HDF5 file.')
parser.add_argument('--fake_images', dest='fake_images_hdf5', required=True, help='Path to the fake images HDF5 file.')
parser.add_argument('--num_neigh', dest='num_neigh', required=True, type=int, help='Number of nearest neighbors to show.')
parser.add_argument('--selected_list', dest='selected_list', required=False, default=None, nargs='+', type=int, help='You can provide a list of generated image indeces to fine its neighbors.')
args = parser.parse_args()
out_path = args.out_path
real_features_hdf5 = args.real_features_hdf5
gen_features_hdf5 = args.gen_features_hdf5
real_images_hdf5 = args.real_images_hdf5
fake_images_hdf5 = args.fake_images_hdf5
num_neigh = args.num_neigh
selected_list = args.selected_list


# with tf.device("/cpu:0"):
save_path = '%s/nearest_neighbors_min_distance.png' % out_path
get_top_nearest_neighbors(num_generated=num_neigh, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, real_img_hdf5=real_images_hdf5, 
						  gen_features_hdf5=gen_features_hdf5, gen_img_hdf5=fake_images_hdf5, maximum=False, random_select=False, save_path=save_path)

save_path = '%s/nearest_neighbors_max_distance.png' % out_path
get_top_nearest_neighbors(num_generated=num_neigh, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, real_img_hdf5=real_images_hdf5, 
						  gen_features_hdf5=gen_features_hdf5, gen_img_hdf5=fake_images_hdf5, maximum=True, random_select=False, save_path=save_path)

save_path = '%s/nearest_neighbors_random.png' % out_path
get_top_nearest_neighbors(num_generated=num_neigh, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, real_img_hdf5=real_images_hdf5, 
						  gen_features_hdf5=gen_features_hdf5, gen_img_hdf5=fake_images_hdf5, maximum=False, random_select=True, save_path=save_path)

if selected_list is not None:
	save_path = '%s/nearest_neighbor_selected.png' % out_path 
	find_top_nearest_neighbors(generated_list=selected_list, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, real_img_hdf5=real_images_hdf5, 
						       gen_features_hdf5=gen_features_hdf5, gen_img_hdf5=fake_images_hdf5, maximum=False, save_path=save_path)