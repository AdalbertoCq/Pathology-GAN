from data_manipulation.preprocessor import Preprocessor



# dataset = 'nki'
# marker = 'he'
# labels = 'nki_survival.csv'

main_path = '/home/adalberto/Documents/Cancer_TMA_Generative'
marker = 'he'
for dataset in ['nki']:
	labels = 'nki_survival.csv'
	for value in [448]:
	    with Preprocessor(patch_h=value, patch_w=value, n_channels=3, dataset=dataset, marker=marker,  labels=labels, overlap=True, save_img=True, threshold=215, project_path=main_path) as preprocessor:
	        preprocessor.run()
