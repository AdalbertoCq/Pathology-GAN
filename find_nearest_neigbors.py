from models.evaluation.tools import *
import argparse

parser = argparse.ArgumentParser(description='PathologyGAN nearest neighbors finder.')
parser.add_argument('--out_path', dest='out_path', required= True, help='Output path folder for images.')
parser.add_argument('--real_features', dest='real_features_hdf5', required= True, help='Path to the real features HDF5 file.')
parser.add_argument('--fake_features', dest='gen_features_hdf5', required= True, help='Path to the fake features HDF5 file.')
parser.add_argument('--num_neigh', dest='batch_size', required= True, type=int, help='Number of nearest neighbors to show.')
args = parser.parse_args()
out_path = args.out_path
real_features_hdf5 = args.real_features_hdf5
gen_features_hdf5 = args.gen_features_hdf5
num_neigh = args.num_neigh
pathgan_type = args.type


with tf.device("/cpu:0"):
	save_path = '%s/nearest_neigbor_min_distance.png' % out_path
	get_top_nearest_neigbors(num_generated=num_neigh, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=False, random_select=False, save_path=save_path)
	
	save_path = '%s/nearest_neigbor_max_distance.png' % out_path
	get_top_nearest_neigbors(num_generated=num_neigh, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=True, random_select=False, save_path=save_path)

	save_path = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/BigGAN/vgh_nki/he/h224_w224_n3/nearest_neigbor_random.png'
	get_top_nearest_neigbors(num_generated=num_neigh, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=False, random_select=True, save_path=save_path)

	save_path = '%s/nearest_neigbor_selected.png' % out_path 
	generated_list = [100, 1131, 1604, 2491, 2700, 2685, 3031, 3035, 3155, 4724]
	find_top_nearest_neigbors(generated_list=generated_list, nearneig=num_neigh, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=False, save_path=save_path)