from models.evaluation.features import *
from data_manipulation.data import Data
import platform

# Common
image_height = 224
image_width = 224
image_channels = 3
batch_size = 50
data_output_path = ''
if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative/'
    main_path = '/media/adalberto/Disk2/Cancer_TMA_Generative/'
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/'

dataset1 = 'nki_vgh'
marker1 = 'he'

dataset2 = 'stanford'
marker2 = 'cd137'
marker2 = 'cathepsin_l'


hdf5s = list()
for percent_data1 in list(range(0,110,10)):
	data1 = Data(dataset=dataset1, marker=marker1, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, labels=False)
	data2 = Data(dataset=dataset2, marker=marker2, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=main_path, labels=False)

	hdf5_path_train, hdf5_path_test = real_samples_contaminated(data1=data1, data2=data2, percent_data1=percent_data1, data_output_path=main_path, num_samples=5000)
	hdf5s.append(hdf5_path_train)
	hdf5s.append(hdf5_path_test)

inception_tf_feature_activations(hdf5s=hdf5s, input_shape=data1.training.shape[1:], batch_size=50, checkpoint_path=None)