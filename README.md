# Pathology-GAN
<!--- Corresponding code of [PathologyGAN](https://arxiv.org/abs/1907.02644) 
 * **'Pathology GAN: Learning deep representations of cancer tissue' Adalberto Claudio Quiros, Roderick Murray-Smith, Ke Yuan. 2019.** --->

**Abstract:**
*We apply Generative Adversarial Networks (GANs) to the domain of digital pathology. Current machine learning research for digital pathology focuses on diagnosis, but we suggest a different approach and advocate that generative models could drive forward the understanding of morphological characteristics of cancer tissue. In this paper, we develop a framework which allows GANs to capture key tissue features and uses these characteristics to give structure to its latent space. To this end, we trained our model on 249K H&E breast cancer tissue images. We show that our model generates high quality images, with a Frechet Inception Distance (FID) of 16.65. We additionally assess the quality of the images with cancer tissue characteristics (e.g. count of cancer, lymphocytes, or stromal cells), using quantitative information to calculate the FID and showing consistent performance of 9.86. Additionally, the latent space of our model shows an interpretable structure and allows semantic vector operations that translate into tissue feature transformations. Furthermore, ratings from two expert pathologists found no significant difference between our generated tissue images from real ones.*

<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/training_tw.gif" width="500">
</p>

## Demo Materials:
* Fake images with the smallest distance to real: For each row, the first image is a generated one, the remaining seven images are close Inception-V1 neighbors of the fake image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/unconditional/neigbor_selected.jpg" width="400">
</p>

* Hand-selected fake images: For each row, the first image is a generated one, the remaining seven images are close Inception-V1 neighbors of the fake image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/unconditional/neighbors_min_dist.jpg" width="400">
</p>

* Individual Images:
<p align="center">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_0.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_1.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_2.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_3.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_4.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_5.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_6.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_7.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_8.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_9.png" width="100"> <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_10.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_11.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_12.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_13.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_14.png" width="100">
<img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/er_negative/individual_images/gen_15.png" width="100">
</p>

* [Images](https://github.com/AdalbertoCq/Pathology-GAN/tree/master/demos/unconditional)
  
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

You can find pre-trained weights for the three different models here:
* [Unconditional](https://figshare.com/s/0a311b5418f21ab2ebd4)
* [Estrogen Receptor](https://figshare.com/s/01c98df16f9c1c01fa3e)
* [Survival Time](https://figshare.com/s/fef199018a1b28ebcd28)

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

* Unconditional:
  * Find the images in the 'evaluation' folder.
```
python3 generate_fake_samples.py --type unconditional --checkpoint ./PathologyGAN_unconditional_weights/PathologyGAN.ckt --num_samples 50
```

* Estrogen Receptor:
  * Find the images in the 'evaluation' folder: er_positive/er_negative
```
python3 generate_fake_samples.py --type er --checkpoint ./PathologyGAN_er_weights/PathologyGAN.ckt --num_samples 50
```
* Survival Time:
  * Find the images in the 'evaluation' folder: survival_positive(>5years)/survival_negative(<=5years).
```
python3 generate_fake_samples.py --type survival --checkpoint ./PathologyGAN_survival_weights/PathologyGAN.ckt --num_samples 50
```

## Training PathologyGAN:
You can find a pre-processed HDF5 file with patches of 224x224x3 resolution [here](https://drive.google.com/open?id=1LpgW85CVA48C8LnpmsDMdHqeCGHKsAxw), each of the patches also contains labeling information of the estrogen receptor status and survival time. Place the 'vgh_nki' under the 'dataset' folder in the main PathologyGAN path.

Each model was trained on an NVIDIA Titan Xp 12 GB for 45 epochs, approximately 72 hours.

```
usage: run_gans.py [-h] --type TYPE [--epochs EPOCHS]
                   [--batch_size BATCH_SIZE]

PathologyGAN trainer.

optional arguments:
  -h, --help               show this help message and exit
  --type TYPE              Type of PathologyGAN: unconditional, er, or survival.
  --epochs EPOCHS          Number epochs to run: default is 45 epochs.
  --batch_size BATCH_SIZE  Batch size, default size is 64.
```

* Unconditional Pathology GAN:
```
python3 run_gans.py --type unconditional
```
* Estrogen Receptor Pathology GAN:
```
python3 run_gans.py --type er
```
* Survival Time Pathology GAN:
```
python3 run_gans.py --type survival
```

