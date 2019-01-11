import pickle
import csv
import os
import skimage.io
import numpy as np


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


# Gets patch from the original image given the config argument:
# Config: _, y, x, rot, flip
# It will also rotate and flip the patch, and returns depeding on norm/flip.
def get_augmented_patch(path, img_filename, config, patch_h=224, patch_w=224, norm=True):
    img_path = os.path.join(path, img_filename)
    img = skimage.io.imread(img_path)
    _, y, x, rot, flip = config
    patch = img[y:y+patch_h, x:x+patch_w]
    rotated = np.rot90(patch, rot)
    flipped = np.fliplr(rotated) if flip else rotated
    return flipped / 255.0 if norm else flipped
