from models.evaluation.tools import *


real_features_hdf5 = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/Real/vgh_nki/he/h224_w224_n3/hdf5_vgh_nki_he_features_train_real.h5'
gen_features_hdf5 = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/BigGAN/vgh_nki/he/h224_w224_n3/hdf5_vgh_nki_he_features_BigGAN.h5'
with tf.device("/cpu:0"):
	save_path = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/BigGAN/vgh_nki/he/h224_w224_n3/nearest_neigbor_min_distance.png'
	get_top_nearest_neigbors(num_generated=10, nearneig=10, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=False, random_select=False, save_path=save_path)

	save_path = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/BigGAN/vgh_nki/he/h224_w224_n3/nearest_neigbor_random.png'
	get_top_nearest_neigbors(num_generated=10, nearneig=10, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=False, random_select=True, save_path=save_path)

	save_path = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/BigGAN/vgh_nki/he/h224_w224_n3/nearest_neigbor_selected.png'
	generated_list = [100, 1131, 1604, 2491, 2700, 2685, 3031, 3035, 3155, 4724]
	find_top_nearest_neigbors(generated_list=generated_list, nearneig=10, real_features_hdf5=real_features_hdf5, gen_features_hdf5=gen_features_hdf5, maximum=False, save_path=save_path)