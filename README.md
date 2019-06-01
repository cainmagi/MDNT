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
    - [ ] Tied and modern transposed convolutional layers for the symmetric autoencoder.
    - [x] Residual layers (or blocks) and their transposed versions.
    - [x] Inception-v4 layers (or blocks) and their transposed versions.
    - [ ] InceptionRes-v2 layers (or blocks) and their transposed versions.
    - [ ] InceptionPlus layers (or blocks) and their transposed versions.
    - [x] External interface for using generic python function.
- [ ] data:
    - [x] Basic h5py IO handles.
    - [ ] Basic HDF5 IO handles.
    - [ ] Basic CSV IO handles.
    - [ ] Basic JSON IO handles.
    - [ ] Data parsing utilities.
- [ ] estimators:
    - [ ] VGG16
    - [ ] U-Net
    - [ ] ResNet
- [ ] functions:
    - [ ] (loss):    Lovasz loss
    - [ ] (metrics): IOU / Jaccard index
    - [ ] (metrics): Pearson correlation coefficient
- [ ] utilities:
    - [ ] Beholder plug-in callback.

## Demos

And we have already written these demos:

The following demos are working for denoising with MNIST data set.

* `demo-AEdense`: Tied autoencoder based on dense layers.
* `demo-normalize`: Normalization layers.
* `demo-AConv`: Modern convolutional layers.
* `demo-AConvTranspose`: Autoencoder based on modern convolutional layers.
* `demo-AConv1DTranspose`: Autoencoder based on modern 1D convolutional layers.
* `demo-ResTranspose`: Autoencoder based on modern residual-v2 layers.
* `demo-InceptTranspose`: Autoencoder based on modern inception-v4 layers.
* `demo-external-1`: External layer test 1. We use numpy-api to convert Cartesian coordinates to polar coordinates.
* `demo-saveH5`: Save `.h5` file.
* `demo-readH5`: Read `.h5` file.
* `demo-readH5Multi`: Read two `.h5` files and combine them as a data set.
* `demo-dataSet`: Use H5 IO to train and test a network.

# Update records

## 0.36 @ 06/01/2019

Finish `Inception1D`, `Inception2D`, `Inception3D`, `Inception1DTranspose`,  `Inception2DTranspose`, `Inception3DTranspose` in `.layers`.

## 0.32 @ 05/31/2019

Finish `Residual1D`, `Residual2D`, `Residual3D`, `Residual1DTranspose`,  `Residual2DTranspose`, `Residual3DTranspose` in `.layers`.

## 0.28 @ 05/24/2019

1. Fix the bug about padding for transposed dilation convolutional layers.
2. Add a new option `output_mshape` to help transposed convolutional layers to control the desired output shape.
3. Finish `PyExternal` in `.layers`.

## 0.24 @ 03/31/2019

Finish `H5GCombiner` in `.data`.

## 0.23 @ 03/27/2019

1. Use `keras.Sequence()` to redefine `H5GParser` and `H5HGParser`.
2. Add compatible check.

## 0.22 @ 03/26/2019

Adjust the `.data.h5py` module to make it more generalized.

## 0.20 @ 03/26/2019

1. Finish `H5HGParser`, `H5SupSaver`, `H5GParser` in `.data`.
2. Finish `DenseTied`, `InstanceNormalization`, `GroupNormalization`, `AConv1D`, `AConv2D`, `AConv3D`, `AConv1DTranspose`,  `AConv2DTranspose`, `AConv3DTranspose` in `.layers`.

## 0.10 @ 03/23/2019

Create this project.