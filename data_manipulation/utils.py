import pickle
import csv
import os
import skimage.io
import numpy as np
import sys
import h5py
import math
import matplotlib.pyplot as plt


def save_image(img, job_id, name, train=True):
    if train:
        folder = 'img'
    else:
        folder = 'gen'
    if not os.path.isdir('run/%s/%s/' % (job_id, folder)):
        os.makedirs('run/%s/%s/' % (job_id, folder))
    skimage.io.imsave('run/%s/%s/%s.png' % (job_id, folder, name), img)


def store_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_csv(file_path):
    with open(file_path, 'r') as file:
        return list(csv.reader(file))


def filter_filenames(filenames, extension):
    return list(filter(lambda f: f.endswith(extension), filenames))


def labels_to_binary(labels, n_bits=5, buckets=True):
    if buckets:
        lower = (labels<=5)*1
        upper = (labels>5)*2
        labels = lower + upper

    labels = labels.astype(int)
    batch_size, l_dim = labels.shape
    output_labels =  np.zeros((batch_size, n_bits))
    for b_num in range(batch_size):
        l = labels[b_num, 0]
        binary_l = '{0:b}'.format(l)
        binary_l = list(binary_l)
        binary_l = list(map(int, binary_l))
        n_rem = n_bits - len(binary_l)
        if n_rem > 0:
            pad =  np.zeros((n_rem), dtype=int)
            pad = pad.tolist()
            binary_l = pad + binary_l
        output_labels[b_num, :] = binary_l
    return output_labels

def survival_5(labels):
    new_l = np.zeros_like(labels)
    upper = (labels>5)*1
    new_l += upper

    return new_l

def labels_to_int(labels):
    batch_size, l_dim = labels.shape
    output_labels =  np.zeros((batch_size, 1))
    line = list()
    for ind in range(l_dim):
        line.append(2**ind)
    line = list(reversed(line))
    line = np.array(line)
    for ind in range(batch_size):
        l = labels[ind, :]
        l_int = int(np.sum(np.multiply(l,line)))
        output_labels[ind, :] = l_int
    return output_labels

def labels_normalize(labels, norm_value=50):
    return labels/norm_value


# Gets patch from the original image given the config argument:
# Config: _, y, x, rot, flip
# It will also rotate and flip the patch, and returns depeding on norm/flip.
def get_augmented_patch(path, img_filename, config, patch_h, patch_w, norm=True):
    img_path = os.path.join(path, img_filename)
    img = skimage.io.imread(img_path)
    _, y, x, rot, flip = config
    patch = img[y:y+patch_h, x:x+patch_w]
    rotated = np.rot90(patch, rot)
    flipped = np.fliplr(rotated) if flip else rotated
    return flipped / 255.0 if norm else flipped


def get_and_save_patch(augmentations, sets, hdf5_path, dataset_path, train_path, patch_h, patch_w, n_channels, type_db, save):
    total = len(augmentations)
    hdf5_file = h5py.File(hdf5_path, mode='w')
    img_db_shape = (total, patch_h, patch_w, n_channels)
    _, label_sample = sets[0]
    if not isinstance(label_sample, (list)):
        len_label = 1
    else:
        len_label = len(label_sample)
    labels_db_shape = (total, len_label)
    img_storage = hdf5_file.create_dataset(name='%s_img' % type_db, shape=img_db_shape, dtype=np.uint8)
    label_storage = hdf5_file.create_dataset(name='%s_labels' % type_db, shape=labels_db_shape, dtype=np.float32)

    print('\nTotal images: ', total)
    index_patches = 0
    for i, patch_config in enumerate(augmentations):
        # Update on progress.
        if i%100 == 0:
            sys.stdout.write('\r%d%% complete  Images processed: %s' % ((i * 100)/total, i))
            sys.stdout.flush()
        index_set, y, x, rot, flip = patch_config
        file_name, labels = sets[index_set]
        try:
            augmented_patch = get_augmented_patch(dataset_path, file_name, patch_config, patch_h, patch_w, norm=False)
        except:
            print('\nCan\'t read image file ', file_name)

        if save:
            label = ''
            if not isinstance(label_sample, (list)):
                label = str(labels)
            else:
                for l in labels:
                    label += '_' + str(l).replace('.', 'p')

            new_file_name = '%s_y%s_x%s_r%s_f%s_label%s.jpg' % (file_name.replace('.jpg', ''), y, x, rot, flip, label)
            new_file_path = os.path.join(train_path, new_file_name)
            skimage.io.imsave(new_file_path, augmented_patch)

        img_storage[index_patches] = augmented_patch
        label_storage[index_patches] = np.array(labels)
        
        index_patches += 1
    hdf5_file.close()
    print()


def make_arrays(train_images, test_images, train_labels, test_labels, patch_h, patch_w, n_channels):
    n_train = len(train_images)
    n_test = len(test_images)
    train_img_data = np.zeros((n_train, patch_h, patch_w, n_channels), dtype=np.uint8)
    train_label_data = np.zeros(n_train, dtype=np.float32)
    test_img_data = np.zeros((n_test, patch_h, patch_w, n_channels), dtype=np.uint8)
    test_label_data = np.zeros(n_test, dtype=np.float32)

    for i in range(n_train):
        train_img_data[i] = train_images[i]
        train_label_data[i] = train_labels[i]
    for i in range(n_test):
        test_img_data[i] = test_images[i]
        test_label_data[i] = test_labels[i]

    return train_img_data, train_label_data, test_img_data, test_label_data


def write_img_data(img_data, patch_h, patch_w, file_name):
    header = np.array([0x0803, len(img_data), patch_h, patch_w], dtype='>i4')
    with open(file_name, "wb") as f:
        f.write(header.tobytes())
        f.write(img_data.tobytes())


def write_label_data(label_data, file_name):
    header = np.array([0x0801, len(label_data)], dtype='>i4')
    with open(file_name, "wb") as f:
        f.write(header.tobytes())
        f.write(label_data.tobytes())


def write_sprite_image(data, filename=None, metadata=True, row_n=None):

    if metadata:
        with open(filename.replace('gen_sprite.png', 'metadata.tsv'),'w') as f:
            f.write("Index\tLabel\n")
            for index in range(data.shape[0]):
                f.write("%d\t%d\n" % (index,index))

    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)
    
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    
    if filename is not None:
        plt.imsave(filename, data)

    return data

def read_hdf5(path, dic):
    hdf5_file = h5py.File(path, 'r')
    return hdf5_file[dic]

