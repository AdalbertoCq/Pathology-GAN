import os
import random
import preparation.utils as utils
import skimage.io


class Preprocessor:
    def __init__(self, project_path=os.getcwd()):
        # Directories handling.
        relative_dataset_path = os.path.join('data', 'nki_he')
        self.dataset_path = os.path.join(project_path, relative_dataset_path)
        self.sets_file_path = os.path.join(self.dataset_path, 'sets')
        self.augmentations_file_path = os.path.join(self.dataset_path, 'augmentations')
        
        # Loading labels.
        label_file = os.path.join(self.dataset_path, 'nki_survival.csv')
        table = utils.load_csv(label_file)
        self.label_dict = self.get_label_dict(table)
        
        # Loading jpg files.
        filenames = os.listdir(self.dataset_path)
        self.image_filenames = utils.filter_filenames(filenames, extension='.jpg')

        self.pathces_per_file = dict()
        

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
    def get_label_dict(table, id_col='rosid', label_col='Survival_2005'):
        # Row defining attributes.
        first_row = table[0]
        # Indexes for the 'rosid' and 'Survival_2005'.
        id_col = first_row.index(id_col)
        label_col = first_row.index(label_col)
        # map(function, iterables)
        # Dictionary with values of key=rosid, value= Survival_2005,
        return dict(map(lambda row: (row[id_col], row[label_col]), table[1:]))

    # 
    def get_label(self, filename):
        patient_id = filename.split('_')[0]
        return float(self.label_dict[patient_id])

    # 
    def append_labels(self, sets):
        return list(map(lambda s: list(map(lambda f: (f, self.get_label(f)), s)), sets))

    # Sets the threshold for the patch, if more than 30% white, discarted.
    @staticmethod
    def satisfactory(img, threshold=215):
        return ((img > threshold).sum(2) == 3).sum() / (img.shape[0] * img.shape[1]) < 0.3

    # Per filename 
    def sample_random_patches(self, filename, n, patch_h=224, patch_w=224):
        
        patches = set()
        attempts = 0
        display = False

        # Keep creating patches, original, rotated and flipped,
        # until you have more than 100 patches or 100 attemps, and
        # length of paths is less than 
        while len(patches) < n // 8 and (patches or attempts < 100):
            attempts += 1
            img = skimage.io.imread(os.path.join(self.dataset_path, filename))
            y = random.randrange(0, img.shape[0] - patch_h + 1)
            x = random.randrange(0, img.shape[1] - patch_w + 1)
            pos = (y, x)
            # Gets patch, flipped horizontally but not rotated or normalized.
            patch = utils.get_augmented_patch(self.dataset_path, filename, (None,) + pos + (0, 0), norm=False)
            if display:
                print(filename, pos, self.satisfactory(patch))
                import matplotlib.pyplot as plt
                plt.imshow(patch)
                plt.show()
            # Make sure that the patch wasn't created at this positition before and that it goes above the threshold. 
            # For each of patch, 4 rotations and 2 flips per rotation.
            if pos not in patches and self.satisfactory(patch):
                patches.add(pos)
                for rot in range(4):
                    for flip in range(2):
                        yield pos + (rot, flip)

    # 
    # sets          [ [file1, file2, ...]   , [file1, file2, ...]   ]
    # augmentations [ [file1, file2, ...]   , [file1, file2, ...]   ]
    # Fixed number of samples:
    # 125 (125 patches/original images)  
    # 8 (Per patch/4 right angle rotations - 2 Horizontal flips per each rotations)
    def augment(self, sets, augmentations, samples=8*125):
        # For train and test lists.
        for set_, augmentation_set in zip(sets, augmentations):
            # If per image file there's no 1000 patch images.
            # Number of files for the set * samples.
            while len(augmentation_set) != len(set_) * samples:
                n = len(augmentation_set)
                # Floor division: changing every 1000=samples, change file after 1000 samples.
                current_img_idx = n // samples
                filename = set_[current_img_idx][0]

                # How good is this? Is it balanced? [VERIFIED]
                # 1000 patches per image.
                # samples - n % samples = remaining samples to get to 1000.
                # it produces samples until this number.
                for config in self.sample_random_patches(filename, samples - n % samples):
                    config = (current_img_idx,) + config
                    augmentation_set.append(config)
                    if filename not in self.pathces_per_file:
                        self.pathces_per_file[filename] = 0    
                    self.pathces_per_file[filename] += 1
                    print(config, len(augmentation_set))

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
            augmentations = [[] for _ in range(n_of_sets)]
            print('"augmentations" not found')
        return augmentations

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def print_patches_balance(self):
        print('Total images used: %s' % len(self.pathces_per_file))
        for filename in self.pathces_per_file:
            print('%s: %s patches' % (filename, self.pathces_per_file[filename]))

    def run(self):
        # Getting lists of images for train and test.
        #   Train                 , Test
        # [ [file1, file2, ...]   , [file1, file2, ...]   ]
        sets_with_labels = self.get_sets()

        # Gets augmentations if available, empty lists otherwise.
        #   Train                 , Test
        # [ [file1, file2, ...]   , [file1, file2, ...]   ]
        augmentations = self.get_augmentations(len(sets_with_labels))

        # Creates augmentation patches from the original images, it stores the information
        # into pickle files.
        self.augment(sets_with_labels, augmentations)

        # Randomize patches for each of the train/test sets.
        for set_ in augmentations:
            random.shuffle(set_)

        # Saves augmentations into a pickle file.
        utils.store_data(augmentations, self.augmentations_file_path)
        print('"augmentations" file finished')

        # print('Balance between patches and files')
        # self.print_patches_balance()
