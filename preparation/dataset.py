import numpy as np
import preparation.utils as utils


class Dataset:
    def __init__(self, path, set_, augmented, batch_size=50, thresholds=()):
        self.set = set_
        self.augmented = augmented
        self.i = 0
        self.path = path
        self.batch_size = batch_size
        self.done = False
        self.thresholds = thresholds

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(self.batch_size)

    @property
    def shape(self):
        return [len(self.augmented), 224, 224, 3]

    def set_pos(self, i):
        self.i = i

    def get_pos(self):
        return self.i

    def reset(self):
        self.set_pos(0)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def adapt_label(self, label):
        thresholds = self.thresholds + (None,)
        adapted = [0.0 for _ in range(len(thresholds))]
        i = None
        for i, threshold in enumerate(thresholds):
            if threshold is None or label < threshold:
                break
        adapted[i] = label if len(adapted) == 1 else 1.0
        return adapted

    def get_example(self, config):
        img_filename, label = self.set[config[0]]
        augmented_patch = utils.get_augmented_patch(self.path, img_filename, config)
        return augmented_patch, self.adapt_label(label), config[0]

    def next_batch(self, n):
        if self.done:
            self.done = False
            raise StopIteration
        examples = list(map(lambda c: self.get_example(c), self.augmented[self.i:self.i + n]))
        self.i += len(examples)
        delta = n - len(examples)
        if delta == n:
            raise StopIteration
        if 0 < delta:
            examples += list(map(lambda c: self.get_example(c), self.augmented[:delta]))
            self.i = delta
            self.done = True
        return tuple(map(np.array, zip(*examples)))
