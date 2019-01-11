import os
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import preparation.utils as utils


dataset_path = '/home/user/project/data/nki_he/'


def show():
    sets_path = os.path.join(dataset_path, 'sets')
    sets = utils.load_data(sets_path)
    train_arr = np.array(list(map(lambda x: x[1], sets[0])))
    test_arr = np.array(list(map(lambda x: x[1], sets[1])))
    bin_hist = np.histogram(train_arr, bins=[0, 10, 30])[0]
    print(bin_hist, bin_hist / bin_hist.sum(), 'train bin')
    multi_hist = np.histogram(train_arr, bins=[0, 5, 10, 15, 30])[0]
    print(multi_hist, multi_hist / multi_hist.sum(), 'train multi')
    bin_hist = np.histogram(test_arr, bins=[0, 10, 30])[0]
    print(bin_hist, bin_hist / bin_hist.sum(), 'test bin')
    multi_hist = np.histogram(test_arr, bins=[0, 5, 10, 15, 30])[0]
    print(multi_hist, multi_hist / multi_hist.sum(), 'test multi')
    all_imgs_with_labels = list(reduce(list.__add__, sets))
    labels = list(map(lambda x: x[1], all_imgs_with_labels))
    arr = np.array(labels)
    bin_hist = np.histogram(arr, bins=[0,10,30])[0]
    print(bin_hist, bin_hist / bin_hist.sum(), 'total bin')
    multi_hist = np.histogram(arr, bins=[0, 5, 10, 15, 30])[0]
    print(multi_hist, multi_hist / multi_hist.sum(), 'total multi')
    median = sorted(labels)[len(labels) // 2]
    print('min:', arr.min(), 'mean:', arr.mean(), 'std:', arr.std(), 'median:', median, 'max:', arr.max())
    plt.hist(labels, bins=range(23), rwidth=0.95)
    plt.xlabel('Survival time (years)', fontsize=16)
    plt.ylabel('Number of images', fontsize=16)
    # plt.title('Survival time distribution')
    plt.show()


if __name__ == '__main__':
    show()