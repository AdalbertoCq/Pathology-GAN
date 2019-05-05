from data_manipulation.preprocessor import Preprocessor
import platform

if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative'
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative'

main_path = '/media/adalberto/Disk2/Cancer_TMA_Generative'

# dataset = 'nki'
# marker = 'he'
# labels = 'nki_survival.csv'

marker = 'cathepsin_l'
for dataset in ['stanford']:
	labels = None
	for value in [224]:
	    with Preprocessor(patch_h=value, patch_w=value, n_channels=3, dataset=dataset, marker=marker,  labels=labels, overlap=True, save_img=True, threshold=240, project_path=main_path) as preprocessor:
	        preprocessor.run()