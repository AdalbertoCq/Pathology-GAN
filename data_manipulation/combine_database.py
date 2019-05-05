from data_manipulation.utils import *
import random
import h5py

nki_path = '/media/adalberto/Disk2/Cancer_TMA_Generative/dataset/nki/he/quarentine/patches_h224_w224'
vgh_path = '/media/adalberto/Disk2/Cancer_TMA_Generative/dataset/vgh/he/quarentine/patches_h224_w224'

hdf_path_nki_train = '%s/hdf5_nki_he_train.h5' % nki_path
hdf_path_nki_test = '%s/hdf5_nki_he_test.h5' % nki_path


hdf_path_vgh_train = '%s/hdf5_vgh_he_train.h5' % vgh_path
hdf_path_vgh_test = '%s/hdf5_vgh_he_test.h5' % vgh_path

train_perc = .8

# Read NKI Files
nki_train = read_hdf5(hdf_path_nki_train, 'train_img')
nki_test = read_hdf5(hdf_path_nki_test, 'test_img')
n_nki_train = len(nki_train)
n_nki_test = len(nki_test)

# Randomize the pointers for NKI.
nki_train_ran_ind = list(range(n_nki_train))
nki_test_ran_ind = list(range(n_nki_test))
random.shuffle(nki_train_ran_ind)
random.shuffle(nki_test_ran_ind)

# Read VGH Files
vgh_train = read_hdf5(hdf_path_vgh_train, 'train_img')
vgh_test = read_hdf5(hdf_path_vgh_test, 'test_img')
n_vgh_train = len(vgh_train)
n_vgh_test = len(vgh_test)

# Randomize the pointers for VGH.
vgh_train_ran_ind = list(range(n_vgh_train))
vgh_test_ran_ind = list(range(n_vgh_test))
random.shuffle(vgh_train_ran_ind)
random.shuffle(vgh_test_ran_ind)

# Combined data 
n_nki_comb_train_1 = int(train_perc*n_nki_train)
n_vgh_comb_train_1 = int(train_perc*n_vgh_train)
n_nki_comb_train_2 = int(train_perc*n_nki_test)
n_vgh_comb_train_2 = int(train_perc*n_vgh_test)
total_train = n_nki_comb_train_1 + n_vgh_comb_train_1 + n_nki_comb_train_2 + n_vgh_comb_train_2

# Combined data 
n_nki_comb_test_1 = n_nki_train-n_nki_comb_train_1
n_vgh_comb_test_1 = n_vgh_train-n_vgh_comb_train_1
n_nki_comb_test_2 = n_nki_test-n_nki_comb_train_2
n_vgh_comb_test_2 = n_vgh_test-n_vgh_comb_train_2
total_test = n_nki_comb_test_1 + n_vgh_comb_test_1 + n_nki_comb_test_2 + n_vgh_comb_test_2

# Train HDF5.
hdf_path_nki_vgh_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/dataset/nki_vgh/he/quarentined/patches_h224_w224/hdf5_nki_vgh_he_train.h5'
hdf5_nki_vgh_train_file = h5py.File(hdf_path_nki_vgh_train, mode='w')
img_db_shape = [total_train] + list(nki_train.shape[1:])
train_img_storage = hdf5_nki_vgh_train_file.create_dataset(name='train_img', shape=img_db_shape, dtype=np.uint8)

ind = 0
for i in nki_train_ran_ind[:n_nki_comb_train_1]:
    train_img_storage[ind] = nki_train[i]
    ind += 1
for i in vgh_train_ran_ind[:n_vgh_comb_train_1]:
    train_img_storage[ind] = vgh_train[i]
    ind += 1
for i in nki_test_ran_ind[:n_nki_comb_train_2]:
    train_img_storage[ind] = nki_test[i]
    ind += 1
for i in vgh_test_ran_ind[:n_vgh_comb_train_2]:
    train_img_storage[ind] = vgh_test[i]
    ind += 1

    
# Train HDF5.
hdf_path_nki_vgh_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/dataset/nki_vgh/he/quarentined/patches_h224_w224/hdf5_nki_vgh_he_test.h5'
hdf5_nki_vgh_test_file = h5py.File(hdf_path_nki_vgh_test, mode='w')
img_db_shape = [total_test] + list(nki_train.shape[1:])
test_img_storage = hdf5_nki_vgh_test_file.create_dataset(name='test_img', shape=img_db_shape, dtype=np.uint8)
ind = 0
for i in nki_train_ran_ind[n_nki_comb_train_1:]:
    test_img_storage[ind] = nki_train[i]
    ind += 1
for i in vgh_train_ran_ind[n_vgh_comb_train_1:]:
    test_img_storage[ind] = vgh_train[i]
    ind += 1
for i in nki_test_ran_ind[n_nki_comb_train_2:]:
    test_img_storage[ind] = nki_test[i]
    ind += 1
for i in vgh_test_ran_ind[n_vgh_comb_train_2:]:
    test_img_storage[ind] = vgh_test[i]
    ind += 1