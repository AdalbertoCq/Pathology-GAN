from data_manipulation.combine_databases import Combine_databases
import platform

if platform.system() == 'Linux':
    main_path = '/home/adalberto/Documents/Cancer_TMA_Generative'
    main_path = '/media/adalberto/Disk2/Cancer_TMA_Generative'
elif platform.system() == 'Darwin':
    main_path = '/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative'

marker = 'he'


for value in [224]:
    with Combine_databases(patch_h=value, patch_w=value, n_channels=3, datasets=['vgh', 'nki'], marker=marker, save_img=True, project_path=main_path) as preprocessor:
    	preprocessor.run()