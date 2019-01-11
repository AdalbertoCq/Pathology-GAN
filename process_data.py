from preparation.preprocessor import Preprocessor

for value in [28]:
    with Preprocessor(patch_h=value, patch_w=value, n_channels=3) as preprocessor:
        preprocessor.run()