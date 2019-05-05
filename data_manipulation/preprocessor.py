import os
import random
import data_manipulation.utils as utils
import skimage.io
import matplotlib.pyplot as plt

class Preprocessor:
    def __init__(self,  patch_h, patch_w, n_channels, dataset, marker,  labels=None, overlap=False, save_img=False, threshold=215, project_path=os.getcwd()):

        # patches size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels

        # Directories and file name handling.
        self.dataset_base = dataset
        self.marker = marker
        self.dataset_name = '%s_%s' % (self.dataset_base, self.marker)
        relative_dataset_path = os.path.join('dataset', self.dataset_base)
        relative_dataset_path = os.path.join(relative_dataset_path, self.marker)
        self.dataset_path = os.path.join(project_path, relative_dataset_path)
        self.sets_file_path = os.path.join(self.dataset_path, 'sets_h%s_w%s' % (patch_h, patch_w))
        self.augmentations_file_path = os.path.join(self.dataset_path, 'augmentations_h%s_w%s' % (patch_h, patch_w))
        self.pathes_path = os.path.join(self.dataset_path, 'patches_h%s_w%s' % (patch_h, patch_w))

        self.hdf5_train = os.path.join(self.pathes_path, 'hdf5_%s_train.h5' % self.dataset_name)
        self.hdf5_test = os.path.join(self.pathes_path, 'hdf5_%s_test.h5' % self.dataset_name)
        self.hdf5 = [self.hdf5_train, self.hdf5_test]

        # Loading labels.
        self.labels = labels

        # Filtering for white percentage.
        self.threshold = threshold

        if self.labels is not None:
            label_file = os.path.join(self.dataset_path, self.labels)
            table = utils.load_csv(label_file)
            if 'nki_survival' in labels:
                id_col='rosid'
                label_col=['Survival_2005', 'ER']
            elif 'vgh_survival' in labels:
                id_col='Patient ID'
                label_col=['Overall Survival', 'ER (IHC)^^'] 
            self.label_dict = self.get_label_dict(table, id_col=id_col, label_col=label_col)

        # Loading jpg files.
        filenames = os.listdir(self.dataset_path)
        self.image_filenames = utils.filter_filenames(filenames, extension='.jpg')

        self.patches_per_file = dict()
        self.save_img = save_img
        self.overlap = overlap

    # Returns lists of images for train/test given the provided shares.
    def split_data_into_sets(self, shares):
        data = self.image_filenames
        assert sum(shares) == 1
        sets = []
        random.shuffle(data)
        start = 0
        for share in shares:
            end = start + round(share * len(data))
            sets.append(data[start:end])
            start = end
        return sets

    # 
    @staticmethod
    def get_label_dict(table, id_col, label_col):
        complete = dict()
        # Row defining attributes.
        first_row = table[0]
        
        # Get indexes of fields in row.
        id_col = first_row.index(id_col)
        l_index = list()
        for l_col in label_col:
            l_index.append(first_row.index(l_col))

        # Get all labels per row.
        for row in table[1:]:
            row_flag = False
            patient = row[id_col]
            row_labels = list()
            for l_i in l_index:
                if '1' == row[l_i] or 'Positive' == row[l_i]:
                    row_labels.append(1)
                    continue
                elif '0' == row[l_i] or 'Negative' == row[l_i]:
                    row_labels.append(0)
                    continue
                row_labels.append(float(row[l_i]))
            complete[patient] = row_labels
        return complete

    # 
    def get_label(self, filename):
        if self.labels is not None:
            patient_id = filename.split('_')[0]
            label = [l for l in self.label_dict[patient_id]]
        else:
            label = filename.split('_')[0]
            label = float(label)
        return label

    # 
    def append_labels(self, sets):
        return list(map(lambda s: list(map(lambda f: (f, self.get_label(f)), s)), sets))

    # Sets the threshold for the patch, if more than 30% white, discarted.
    @staticmethod
    def satisfactory(img, threshold=240):
        return ((img > threshold).sum(2) == 3).sum() / (img.shape[0] * img.shape[1]) < .3

    # Per filename 
    def sample_patches(self, filename):
        patches = set()
        display = False
        img = skimage.io.imread(os.path.join(self.dataset_path, filename))

        # height -> y, width -> x.
        height, width, channels = img.shape
        # Hack for 224 patches, so there's more.
        num_y = int(height//(self.patch_h/2))
        num_x = int(width//(self.patch_w/2))
        # num_y = height//self.patch_h
        # num_x = width//self.patch_w

        for i_x in range(0, num_x-1):
            for i_y in range(0, num_y-1):
                if self.overlap:
                    y = i_y * int(self.patch_h//2)
                    x = i_x * int(self.patch_w//2)
                else:
                    y = i_y * self.patch_h
                    x = i_x * self.patch_w
                pos = (y, x)
                # Gets patch, flipped horizontally but not rotated or normalized.
                patch = utils.get_augmented_patch(self.dataset_path, filename, (None,) + pos + (0, 0), self.patch_h, self.patch_w, norm=False)
                if display:
                    print(filename, pos, self.satisfactory(patch, self.threshold))
                    import matplotlib.pyplot as plt
                    plt.imshow(patch)
                    plt.show()

                # Make sure that the patch wasn't created at this position before and that it goes above the threshold.
                # For each of patch, 4 rotations and 2 flips per rotation.
                if pos not in patches and self.satisfactory(patch, self.threshold):
                    patches.add(pos)
                    for rot in range(4):
                        for flip in range(2):
                            yield pos + (rot, flip)

    '''
    sets = [set_train, set_test]
    set_train = [('231___2_114_13_6.jpg', 13.4630136986) , ...]
    set_test  = [('223___2_114_13_6.jpg', 41.1231265464) , ...]
    augmentation = []
    '''
    def augment(self, sets, augmentations):
        # For train and test lists.
        for set_, augmentation_set in zip(sets, augmentations):
            '''
            Set: [('231___2_114_13_6.jpg', 13.4630136986) , ...]
                 [('Image name.jpg',       Survival years), ...]
            '''
            current_img_idx = 0
            for filename, labels in set_:
                self.patches_per_file[filename] = 0
                n = len(augmentation_set)
                for config in self.sample_patches(filename):
                    config = (current_img_idx,) + config
                    augmentation_set.append(config)
                    self.patches_per_file[filename] += 1
                print('Patches per image: %s %s ' % (filename, self.patches_per_file[filename]))

                # If no new patches are created, remove image file from list and replace sets pickle.
                # Possibilities?
                # Useless image: most of the picture is white.
                if n == len(augmentation_set):
                    del set_[current_img_idx]
                    utils.store_data(sets, self.sets_file_path)
                    print('"sets" updated (%s discarded)' % filename)

                # Save augmentations in each iteration.
                utils.store_data(augmentations, self.augmentations_file_path)
                print('checkpoint saved')
                current_img_idx += 1

    # Gets lists of train/test images either from pickle file or from the data path.
    def get_sets(self):
        try:
            sets_with_labels = utils.load_data(self.sets_file_path)
            print('"sets" file loaded')
        except FileNotFoundError:
            sets = self.split_data_into_sets([0.8, 0.2])
            sets_with_labels = self.append_labels(sets)
            utils.store_data(sets_with_labels, self.sets_file_path)
            print('"sets" file created')
        return sets_with_labels

    # Gets augmentations if available, empty lists otherwise.
    def get_augmentations(self, n_of_sets):
        try:
            augmentations = utils.load_data(self.augmentations_file_path)
            print('"augmentations" file loaded')
        except FileNotFoundError:
            augmentations = None
            print('"augmentations" not found')
        return augmentations

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def save_images(self, augmentations, sets, save):
        if os.path.isdir(self.pathes_path):
            print('Folder already exists: ', self.pathes_path)
            exit(1)
        os.makedirs(self.pathes_path)

        for index, s in enumerate(['train', 'test']):
            t_path = os.path.join(self.pathes_path, s)
            os.makedirs(t_path)
            utils.get_and_save_patch(augmentations[index], sets[index], self.hdf5[index], self.dataset_path, t_path, self.patch_h, self.patch_w, self.n_channels,
                                     type_db=s, save=save)

    def run(self):
        # Getting lists of images for train and test.
        #   Train                 , Test
        # [ [file1, file2, ...]   , [file1, file2, ...]   ]
        sets_with_labels = self.get_sets()

        # Gets augmentations if available, empty lists otherwise.
        #   Train                 , Test
        # [ [file1, file2, ...]   , [file1, file2, ...]   ]
        augmentations = self.get_augmentations(len(sets_with_labels))

        # If the augmentation reference file already exists, don't run this again.
        if augmentations is None:

            augmentations = [[] for _ in range(len(sets_with_labels))]
            # Creates augmentation patches from the original images, it stores the information
            # into pickle files.
            self.augment(sets_with_labels, augmentations)

            # Randomize patches for each of the train/test sets.
            for set_ in augmentations:
                random.shuffle(set_)

            # Saves augmentations into a pickle file.
            utils.store_data(augmentations, self.augmentations_file_path)
            print('"augmentations" file finished')

        # Creates Hdf5 database and save patches images, saving the patches if off by default.
        self.save_images(augmentations, sets_with_labels, save=self.save_img)