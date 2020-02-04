# Pathology-GAN
<!--- Corresponding code of [PathologyGAN](https://arxiv.org/abs/1907.02644) 
 * **'Pathology GAN: Learning deep representations of cancer tissue' Adalberto Claudio Quiros, Roderick Murray-Smith, Ke Yuan. 2019.** --->

**Abstract:**
*We apply Generative Adversarial Networks (GANs) to the domain of digital pathology. Current machine learning research for digital pathology focuses on diagnosis, but we suggest a different approach and advocate that generative models could drive forward the understanding of morphological characteristics of cancer tissue. In this paper, we develop a framework which allows GANs to capture key tissue features and uses these characteristics to give structure to its latent space. To this end, we trained our model on 249K H&E breast cancer tissue images. We show that our model generates high quality images, with a Frechet Inception Distance (FID) of 16.65. We additionally assess the quality of the images with cancer tissue characteristics (e.g. count of cancer, lymphocytes, or stromal cells), using quantitative information to calculate the FID and showing consistent performance of 9.86. Additionally, the latent space of our model shows an interpretable structure and allows semantic vector operations that translate into tissue feature transformations. Furthermore, ratings from two expert pathologists found no significant difference between our generated tissue images from real ones.*

<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/training_tw.gif" width="500">
</p>

## Demo Materials:

* Latent space images:
  <p align="center">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/latent_space/UMAP_StylePathologyGAN_latent_space_zdim_200_dimension_2.jpg" width="200">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/latent_space/UMAP_PathologyGAN_latent_space_zdim_200_dimension_2.jpg" width="200">
  </p>

  * [Images](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/latent_space)

* Vector operation examples:
  <p align="center">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/vector_op/op_0_72.jpg" width="350">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/vector_op/op_0_58.jpg" width="350">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/vector_op/op_1_0.jpg" width="250">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/vector_op/op_1_1.jpg" width="250">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/vector_op/op_2_76.jpg" width="350">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/vector_op/op_2_78.jpg" width="350">
  </p>

  * [Images](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/vector_op)

* Vector operation examples:
  - Fake images with the smallest distance to real: For each row, the first image is a generated one, the remaining seven images are close Inception-V1 neighbors of the fake image:
  <p align="center">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/neigbor_selected.jpg" width="400">
  </p>

  - Hand-selected fake images: For each row, the first image is a generated one, the remaining seven images are close Inception-V1 neighbors of the fake image:
  <p align="center">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/neighbors_min_dist.jpg" width="400">
  </p>

* Individual Images:
  <p align="center">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_0.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_1.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_2.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_3.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_4.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_5.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_6.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_7.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_8.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_9.png" width="100"> <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_10.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_11.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_12.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_13.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_14.png" width="100">
    <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/individual_images/gen_15.png" width="100">
  </p>

  * [Images](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/)
  
## Datasets:
H&E breast cancer databases from the Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) cohort with 248 and 328 patients respectevely. Each of them include tissue micro-array (TMA) images, along with clinical patient data such as survival time, and estrogen-receptor (ER) status. The original TMA images all have a resolution of 1128x720 pixels, and we split each of the images into smaller patches of 224x224, and allow them to overlap by 50%. We also perform data augmentation on these images, a rotation of 90 degrees, and 180 degrees, and vertical and horizontal inversion. We filter out images in which the tissue covers less than 70% of the area. In total this yields a training set of 249K images, and a test set of 62K.

We use these Netherlands Cancer Institute (NKI) cohort and the Vancouver General Hospital (VGH) previously used in Beck et al. \[1]. These TMA images are from the [Stanford Tissue Microarray Database](https://tma.im/cgi-bin/home.pl)[2]

\[1] Beck, A.H. and Sangoi, A.R. and Leung, S. Systematic analysis of breast cancer morphology uncovers stromal features associated with survival. Science translational medicine (2018).

\[2] Robert J. Marinelli, Kelli Montgomery, Chih Long Liu, Nigam H. Shah, Wijan Prapong, Michael Nitzberg, Zachariah K. Zachariah, Gavin J. Sherlock, Yasodha Natkunam, Robert B. West, Matt van de Rijn, Patrick O. Brown, and Catherine A. Ball. The Stanford Tissue Microarray Database. Nucleic Acids Res 2008 36(Database issue): D871-7. Epub 2007 Nov 7 doi:10.1093/nar/gkm861.

You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time.

This is a sample of an original TMA image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/original_tma.jpg" width="400">
</p>

## Pre-trained Models:

You can find pre-trained weights for the model [here](https://figshare.com/s/0a311b5418f21ab2ebd4)

## Python Enviroment:
```
h5py                    2.9.0
numpy                   1.16.1
pandas                  0.24.1
scikit-image            0.14.2
scikit-learn            0.20.2
scipy                   1.2.0
seaborn                 0.9.0
sklearn                 0.0
tensorboard             1.12.2
tensorflow              1.12.0
tensorflow-probability  0.5.0
python                  3.6.7
```

## Load model and generate images:

* Find the images in the 'evaluation' folder:
```
usage: generate_fake_samples.py [-h] --checkpoint CHECKPOINT
                                [--num_samples NUM_SAMPLES]
                                [--batch_size BATCH_SIZE] --z_dim Z_DIM

PathologyGAN fake image generator.

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to pre-trained weights (.ckt) of PathologyGAN.
  --num_samples NUM_SAMPLES
                        Number of images to generate.
  --batch_size BATCH_SIZE
                        Batch size.
  --z_dim Z_DIM         Latent space size.
```

* Usage example:  
```
python3 ./generate_fake_samples.py --num_samples 100 --batch_size 50 --z_dim 200 --checkpoint data_model_output/PathologyGAN/h224_w224_n3_zdim_200/checkpoints/PathologyGAN.ckt
```

## Training PathologyGAN:
You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time. Place the 'vgh_nki' under the 'dataset' folder in the main PathologyGAN path.

Each model was trained on an NVIDIA Titan Xp 12 GB for 45 epochs, approximately 72 hours.

```
usage: run_pathgan.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                      [--model MODEL]

PathologyGAN trainer.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number epochs to run: default is 45 epochs.
  --batch_size BATCH_SIZE
                        Batch size, default size is 64.
  --model MODEL         Model name.
```

* Pathology GAN training example:
```
python3 run_pathgan.py 
```

## Find closest and furthest neighbors between generated and real images:
* You will need the Inception-V1 features of the images in a HDF5 file format.
```
usage: find_nearest_neighbors.py [-h] --out_path OUT_PATH --real_features
                                REAL_FEATURES_HDF5 --fake_features
                                GEN_FEATURES_HDF5 --num_neigh NUM_NEIGH
                                [--selected_list SELECTED_LIST [SELECTED_LIST ...]]

PathologyGAN nearest neighbors finder.

optional arguments:
  -h, --help            show this help message and exit
  --out_path OUT_PATH   Output path folder for images.
  --real_features REAL_FEATURES_HDF5
                        Path to the real features HDF5 file.
  --fake_features GEN_FEATURES_HDF5
                        Path to the fake features HDF5 file.
  --num_neigh NUM_NEIGH
                        Number of nearest neighbors to show.
  --selected_list SELECTED_LIST [SELECTED_LIST ...]
                        You can provide a list of generated image indeces to
                        fine its neighbors.
```
Usage example:
```
python3 find_nearest_neighbors.py --out_path project_path --real_features project_path/hdf5_nki_vgh_he_features_real.h5 --fake_features project_path/hdf5_nki_vgh_he_features_generated.h5 --num_neigh 10 --selected_list 100 1131 1604 2491 2700 2685 3031 303 3155 4724
```
