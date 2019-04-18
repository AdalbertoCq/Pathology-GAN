import os
import random
import data_manipulation.utils as utils
import skimage.io
import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys

class Combine_databases:
    def __init__(self,  patch_h, patch_w, n_channels, datasets, marker, save_img, project_path=os.getcwd()):

        # patches size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels

        # Directories and file name handling.
        self.datasets = dict()
        for dataset in datasets:
            self.datasets[dataset] = dict()
            dataset_name = '%s_%s' % (dataset, marker)
            relative_dataset_path = os.path.join('dataset', dataset)
            relative_dataset_path = os.path.join(relative_dataset_path, marker)
            dataset_path = os.path.join(project_path, relative_dataset_path)
            self.datasets[dataset]['sets_file'] = os.path.join(dataset_path, 'sets_h%s_w%s' % (patch_h, patch_w))
            self.datasets[dataset]['augmentations_file'] = os.path.join(dataset_path, 'augmentations_h%s_w%s' % (patch_h, patch_w))
            self.datasets[dataset]['dataset_path'] = dataset_path
            filenames = os.listdir(dataset_path)
            self.datasets[dataset]['image_file'] = utils.filter_filenames(filenames, extension='.jpg')
        
        self.total_dataset = '_'.join(datasets)
        relative_dataset_path = os.path.join('dataset', self.total_dataset)
        relative_dataset_path = os.path.join(relative_dataset_path, marker)
        self.dataset_path = os.path.join(project_path, relative_dataset_path)
        self.pathes_path = os.path.join(self.dataset_path, 'patches_h%s_w%s' % (patch_h, patch_w))

        self.hdf5_train = os.path.join(self.pathes_path, 'hdf5_%s_%s_train.h5' % (self.total_dataset, marker))
        self.hdf5_test = os.path.join(self.pathes_path, 'hdf5_%s_%s_test.h5' % (self.total_dataset, marker))
        self.hdf5 = [self.hdf5_train, self.hdf5_test]

        self.save_img = save_img

    def combine_randomize_sets(self):
        augmentations = [list(), list()]
        type_db_augmentations = [list(), list()]
        i_s = [list(), list()]

        for dataset in self.datasets:
            augmentations_dataset = utils.load_data(self.datasets[dataset]['augmentations_file'])
            sets_dataset = utils.load_data(self.datasets[dataset]['sets_file'])
            for i, augmentations_t in enumerate(augmentations_dataset):
                for element in augmentations_t:
                    ind, a, b, c, d = element
                    file_name = sets_dataset[i][ind][0]
                    augmentations[i].append((file_name, a, b, c, d))
                db = [dataset]*len(augmentations_dataset[i])    
                type_db_augmentations[i].extend(db)

        for dataset in self.datasets:
            sets_dataset = utils.load_data(self.datasets[dataset]['sets_file'])
            self.datasets[dataset]['sets'] = dict()
            for i, set_s in enumerate(sets_dataset):
                self.datasets[dataset]['sets'][i] = dict()
                for file_name, labels in set_s:
                    self.datasets[dataset]['sets'][i][file_name] = labels

        for i, augmentations_t in enumerate(augmentations):
            ind = list(range(len(augmentations_t)))
            random.shuffle(ind)
            i_s[i] = ind

        return augmentations, type_db_augmentations, i_s

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def save_images(self, augmentations, type_db_augmentations, i_s, save):
        if os.path.isdir(self.pathes_path):
            print('Folder already exists: ', self.pathes_path)
            exit(1)
        os.makedirs(self.pathes_path)

        for index, s in enumerate(['train', 'test']):
            t_path = os.path.join(self.pathes_path, s)
            os.makedirs(t_path)
            self.get_and_save_patch(augmentations[index], type_db_augmentations[index], i_s[index], self.hdf5[index], self.dataset_path, t_path, self.patch_h, self.patch_w, self.n_channels, type_db=s, i_t=index, save=save)

    def get_and_save_patch(self, augmentations, type_db_augmentations, i_s, hdf5_path, dataset_path, train_path, patch_h, patch_w, n_channels, type_db, i_t, save):
        total = len(augmentations)
        hdf5_file = h5py.File(hdf5_path, mode='w')

        sample = next(iter(self.datasets))
        sets_dataset = utils.load_data(self.datasets[sample]['sets_file'])
        _, label_sample = sets_dataset[0][0]

        img_db_shape = (total, patch_h, patch_w, n_channels)
        labels_db_shape = (total, len(label_sample))        
        img_storage = hdf5_file.create_dataset(name='%s_img' % type_db, shape=img_db_shape, dtype=np.uint8)
        label_storage = hdf5_file.create_dataset(name='%s_labels' % type_db, shape=labels_db_shape, dtype=np.float32)

        print('\nTotal images: ', total)
        index_patches = 0
        for i, index in enumerate(i_s):
            patch_config = augmentations[index]
            dataset = type_db_augmentations[index]
            # Update on progress.
            if i%100 == 0:
                sys.stdout.write('\r%d%% complete  Images processed: %s' % ((i * 100)/total, i))
                sys.stdout.flush()

            file_name, y, x, rot, flip = patch_config
            labels = self.datasets[dataset]['sets'][i_t][file_name]
            dataset_path = self.datasets[dataset]['dataset_path']

            try:
                augmented_patch = utils.get_augmented_patch(dataset_path, file_name, patch_config, patch_h, patch_w, norm=False)
            except:
                print('\nCan\'t read image file ', file_name, ' of dataset', dataset)

            if save:
                label = ''
                for l in labels:
                    label += '_' + str(l).replace('.', 'p')

                new_file_name = '%s_%s_y%s_x%s_r%s_f%s_label%s.jpg' % (dataset, file_name.replace('.jpg', ''), y, x, rot, flip, label)
                new_file_path = os.path.join(train_path, new_file_name)
                skimage.io.imsave(new_file_path, augmented_patch)

            img_storage[index_patches] = augmented_patch
            label_storage[index_patches] = np.array(labels)
            
            index_patches += 1
        hdf5_file.close()
        print()

    def run(self):
        augmentations, type_db_augmentations, i_s = self.combine_randomize_sets()
        self.save_images(augmentations, type_db_augmentations, i_s, self.save_img)


