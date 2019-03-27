# Modern Deep Network Toolkits for Tensorflow-Keras

We proudly present our newest produce, a totally well-defined extension for Tensorflow-Keras users!

## Documentation

Still not available now, will implement in the future.

## Progress

Now we have such progress on the semi-product:

- [ ] optimzers:
    - still working on it.
    - [x] Wrapped default optimizers.
- [ ] layers:
    - [x] Tied dense layer for the symmetric autoencoder.
    - [x] Extened normalization layers.
    - [x] Modern convolutional layers.
    - [x] Modern transposed convolutional layers.
    - [ ]  Tied and modern transposed convolutional layers for the symmetric autoencoder.
    - [ ]  Residual layers (or blocks) and their transposed versions.
    - [ ]  Inception-v4 layers (or blocks) and their transposed versions.
- [ ] data:
    - [x] Basic h5py IO handles.
    - [ ]  Basic HDF5 IO handles.
    - [ ]  Basic CSV IO handles.
    - [ ]  Basic JSON IO handles.
    - [ ]  Data parsing utilities.
- [ ] estimators:
    - [ ] VGG16
    - [ ] U-Net
    - [ ] ResNet
- [ ] functions:
    - [ ] (loss):    Lovasz loss
    - [ ] (metrics): IOU / Jaccard index
    - [ ] (metrics): Pearson correlation coefficient

## Demos

And we have already written these demos:

The following demos are working for denoising with MNIST data set.

* `demo-AEdense`: Tied autoencoder based on dense layers.
* `demo-normalize`: Normalization layers.
* `demo-AConv`: Modern convolutional layers.
* `demo-AConvTranspose`: Autoencoder based on modern convolutional layers.
* `demo-AConv1DTranspose`: Autoencoder based on modern 1D convolutional layers.
* `demo-saveH5`: Save `.h5` file.
* `demo-readH5`: Read `.h5` file.
* `demo-dataSet`: Use H5 IO to train and test a network.

# Update records

## 0.20 @ 03/26/2019

1. Finish `H5HGParser`, `H5SupSaver`, `H5GParser` in `.data`
2. Finish `DenseTied`, `InstanceNormalization`, `GroupNormalization`, `AConv1D`, `AConv2D`, `AConv3D`, `AConv1DTranspose`,  `AConv2DTranspose`, `AConv3DTranspose` in `.layers`.

## 0.10 @ 03/23/2019

Create this project.