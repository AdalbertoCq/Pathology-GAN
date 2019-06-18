# Pathology-GAN

## Demo Materials:
* Fake images with the smallest distance to real: For each row, the first image is a generated one, the remaining seven images are close Inception-V1 neighbors of the fake image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/unconditional/neigbor_selected.png" width="400">
</p>

* Hand-selected fake images: For each row, the first image is a generated one, the remaining seven images are close Inception-V1 neighbors of the fake image:
<p align="center">
  <img src="https://github.com/AdalbertoCq/Pathology-GAN/blob/master/demos/unconditional/neighbors_min_dist.png" width="400">
</p>


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

## Training PathologyGAN:
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


