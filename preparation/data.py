import os
from preparation.utils import load_data
from preparation.dataset import Dataset


class Data:
    def __init__(self, project_path=os.getcwd(), batch_size=50, thresholds=()):
        relative_dataset_path = os.path.join('data', 'nki_he')
        dataset_path = os.path.join(project_path, relative_dataset_path)
        sets_file_path = os.path.join(dataset_path, 'sets')
        augmented_sets_file_path = os.path.join(dataset_path, 'augmentations')
        sets = load_data(sets_file_path)
        augmented = load_data(augmented_sets_file_path)
        self.training = Dataset(dataset_path, sets[0], augmented[0], batch_size=batch_size, thresholds=thresholds)
        self.test = Dataset(dataset_path, sets[1], augmented[1], batch_size=batch_size, thresholds=thresholds)
