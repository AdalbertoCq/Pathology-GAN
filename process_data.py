from preparation.preprocessor import Preprocessor

for value in [28, 56, 128, 256, 512]:
    with Preprocessor(patch_h=value, patch_w=value, n_channels=3, dataset='StanfordTMA', marker='cd81',  labels=False,
                      project_path='/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative') as preprocessor:
        preprocessor.run()