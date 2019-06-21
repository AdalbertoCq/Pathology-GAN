import os
from data_manipulation.dataset import Dataset


class Data:
    def __init__(self, dataset, marker, patch_h, patch_w, n_channels, batch_size, project_path=os.getcwd(), thresholds=(), labels=True, empty=False):

        # Directories and file name handling.
        self.dataset = dataset
        self.marker = marker
        self.dataset_name = '%s_%s' % (self.dataset, self.marker)
        relative_dataset_path = os.path.join(self.dataset, self.marker)
        relative_dataset_path = os.path.join('dataset', relative_dataset_path)
        relative_dataset_path = os.path.join(project_path, relative_dataset_path)
        self.pathes_path = os.path.join(relative_dataset_path, 'patches_h%s_w%s' % (patch_h, patch_w))

        self.hdf5_train = os.path.join(self.pathes_path, 'hdf5_%s_train.h5' % self.dataset_name)
        self.hdf5_test = os.path.join(self.pathes_path, 'hdf5_%s_test.h5' % self.dataset_name)

        # Train dataset
        self.training = Dataset(self.hdf5_train, patch_h, patch_w, n_channels, batch_size=batch_size, data_type='train', thresholds=thresholds, labels=labels, empty=empty)
        # Test dataset
        self.test = Dataset(self.hdf5_test, patch_h, patch_w, n_channels, batch_size=batch_size, data_type='test', thresholds=thresholds, labels=labels, empty=empty)
