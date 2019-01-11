import os
from preparation.dataset import Dataset


class Data:
    def __init__(self, patch_h, patch_w, n_channels, batch_size, project_path=os.getcwd(), thresholds=()):

        # Directories and file name handling.
        relative_dataset_path = os.path.join('dataset', 'nki_he')
        dataset_path = os.path.join(project_path, relative_dataset_path)
        self.pathes_path = os.path.join(relative_dataset_path, 'patches_h%s_w%s' % (patch_h, patch_w))

        self.hdf5_train = os.path.join(self.pathes_path, 'hdf5_nki_he_train.h5')
        self.hdf5_test = os.path.join(self.pathes_path, 'hdf5_nki_he_test.h5')

        # Train dataset
        self.training = Dataset(self.hdf5_train, patch_h, patch_w, n_channels, batch_size=batch_size, data_type='train', thresholds=thresholds)
        # Test dataset
        self.test = Dataset(self.hdf5_test, patch_h, patch_w, n_channels, batch_size=batch_size, data_type='test', thresholds=thresholds)
