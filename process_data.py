from data_manipulation.preprocessor import Preprocessor
import platform

if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative'
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative'

dataset = 'stanford'

for value in [224]:
	for marker in ['vickz', 'nmb', 'cd137']:
	    with Preprocessor(patch_h=value, patch_w=value, n_channels=3, dataset=dataset, marker=marker,  labels=False, overlap=True, save_img=True, project_path=main_path) as preprocessor:
	        preprocessor.run()