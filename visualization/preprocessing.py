import matplotlib.pyplot as plt
import preparation.utils as utils
import os
from skimage.io import imread, imsave


path = os.path.join(os.sep, 'home', 'user', 'project', 'data', 'nki_he')
sets = utils.load_data(os.path.join(path, 'sets'))
augmentations = utils.load_data(os.path.join(path, 'augmentations'))


def show_augmentations_and_filtering(filename='59___3_110_13_2.jpg'):
    set_idx = None
    idx = None
    for i, set_ in enumerate(sets):
        set_ = list(map(lambda x: x[0], set_))
        try:
            idx = set_.index(filename)
            set_idx = i
        except ValueError:
            pass
    rows = 3
    columns = 4
    _, ax = plt.subplots(rows, columns)
    patches = list(filter(lambda x: x[0] == idx, augmentations[set_idx]))
    samples = patches[:rows*columns]
    for i in range(rows):
        for j in range(columns):
            config = samples[columns * i + j]
            patch = utils.get_augmented_patch(path, filename, config)
            ax[i][j].imshow(patch)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()
    img = imread(os.path.join(path, filename))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    threshold = 215
    proc = ((img > threshold).sum(2) == 3).sum() / (img.shape[0] * img.shape[1])
    print(proc)
    mask = (img > threshold).sum(2, keepdims=True) == 3
    import numpy as np
    mask = np.tile(mask, [1, 1, 3])
    img[mask] = 0
    # imsave('/home/user/project/filtering.png', img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


show_augmentations_and_filtering()
