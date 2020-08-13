import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from collections import OrderedDict
from models.score.utils import *


def get_top_nearest_neighbors(num_generated, nearneig, real_features_hdf5, real_img_hdf5, gen_features_hdf5, gen_img_hdf5, maximum=False, random_select=False, save_path=None):

    real_features_file = h5py.File(real_features_hdf5, 'r')
    gen_features_file = h5py.File(gen_features_hdf5, 'r')
    real_img_file = h5py.File(real_img_hdf5, 'r')
    gen_img_file = h5py.File(gen_img_hdf5, 'r')

    print(real_img_file.keys())
    real_features = real_features_file['features']
    gen_features = gen_features_file['features']
    real_img = real_img_file['images']
    gen_img = gen_img_file['images']

    with tf.Session() as sess:
        real_features = tf.constant(np.array(real_features), dtype=tf.float32)
        gen_features = tf.constant(np.array(gen_features), dtype=tf.float32)

        # Get Nearest Neighbors for all generated images.
        gen_real_distances = tf.sqrt(tf.abs(euclidean_distance(gen_features, real_features)))
        neg = tf.negative(gen_real_distances)
        neg_s_distances, s_indices = tf.math.top_k(input=neg, k=nearneig, sorted=True)
        s_distances = tf.negative(neg_s_distances)


        # Getting the top smallest distances between Generated and Real images.
        neg_s_distances1, s_indices1 = tf.math.top_k(input=neg, k=1, sorted=True)
        neg_s_distances1 = tf.transpose(neg_s_distances1)
        if not random_select:
            if maximum:
                neg_s_distances1 = tf.negative(neg_s_distances1)
            neg_s_distances1, s_indices1 = tf.math.top_k(input=neg_s_distances1, k=num_generated, sorted=True)
            s_indices1 = tf.transpose(s_indices1)
            s_indices1 = s_indices1.eval()
        else:
            lin = list(range(int(gen_real_distances.shape[0])))
            random.shuffle(lin)
            s_indices1 = np.zeros((num_generated,1), dtype=np.int8)
            s_indices1[:, 0] = lin[:num_generated]
            
        s_indices = s_indices.eval()
        s_distances = s_distances.eval()
        # For the images with top smallest distances, show nearest neighbors.
        neighbors = OrderedDict()
        for i, ind in enumerate(s_indices1):
            ind = ind[0]
            neighbors[ind] = list() 
            for j in range(nearneig):
                neighbors[ind].append((s_indices[ind,j], s_distances[ind,j]))

        if save_path is not None:
            height, width, channels = real_img.shape[1:]
            grid = np.zeros((num_generated*height, (nearneig+1)*width, channels))
            for i, ind in enumerate(s_indices1):
                ind = ind[0]
                total = gen_img[ind]
                for j in range(nearneig):
                    real = real_img[s_indices[ind,j]]/255.
                    total = np.concatenate([total, real], axis=1)
                grid[i*height:(i+1)*height, :, :] = total
            plt.imsave(save_path, grid)

        return neighbors


def find_top_nearest_neighbors(generated_list, nearneig, real_features_hdf5, real_img_hdf5, gen_features_hdf5, gen_img_hdf5, maximum=False, save_path=None):
    real_features_file = h5py.File(real_features_hdf5, 'r')
    gen_features_file = h5py.File(gen_features_hdf5, 'r')
    real_img_file = h5py.File(real_img_hdf5, 'r')
    gen_img_file = h5py.File(gen_img_hdf5, 'r')

    real_features = real_features_file['features']
    gen_features = gen_features_file['features']
    real_img = real_img_file['images']
    gen_img = gen_img_file['images']

    with tf.Session() as sess:
        real_features = tf.constant(np.array(real_features), dtype=tf.float32)
        gen_features = tf.constant(np.array(gen_features), dtype=tf.float32)

        # Get Nearest Neighbors for all generated images.
        gen_real_distances = tf.sqrt(tf.abs(euclidean_distance(gen_features, real_features)))
        neg = tf.negative(gen_real_distances)
        neg_s_distances, s_indices = tf.math.top_k(input=neg, k=nearneig, sorted=True)
        s_distances = tf.negative(neg_s_distances)

        s_indices = s_indices.eval()
        s_distances = s_distances.eval()
        # For the images with top smallest distances, show nearest neighbors.
        height, width, channels = real_img.shape[1:]
        neighbors = dict()
        grid = np.zeros((len(generated_list)*height, (nearneig+1)*width, channels))
        for i, ind in enumerate(generated_list):
            total = gen_img[ind]
            neighbors[ind] = list() 
            for j in range(nearneig):
                neighbors[ind].append((s_indices[ind,j], s_distances[ind,j]))
                real = real_img[s_indices[ind,j]]/255.
                total = np.concatenate([total, real], axis=1)
            grid[i*height:(i+1)*height, :, :] = total
        plt.imshow(grid)
        if save_path is not None:
            plt.imsave(save_path, grid)
        return neighbors




    
