# Modern Deep Network Toolkits for Tensorflow-Keras

We proudly present our newest produce, a totally well-defined extension for Tensorflow-Keras users!

## Documentation

Still not available now, will implement in the future.

## Progress

Now we have such progress on the semi-product:

- [ ] optimzers:
    - still working on it.
    - [x] Wrapped default optimizers.
- [x] layers:
    - [x] Tied dense layer for the symmetric autoencoder.
    - [x] Extended dropout and noise layers.
    - [x] Extened normalization layers.
    - [x] Group convolutional layers.
    - [x] Modern convolutional layers (support group convolution).
    - [x] Modern transposed convolutional layers (support group convolution).
    - [x] Tied (trivial) transposed convolutional layers for the symmetric autoencoder.
    - [x] Residual layers (or blocks) and their transposed versions.
    - [x] ResNeXt layers (or blocks) and their transposed versions.
    - [x] Inception-v4 layers (or blocks) and their transposed versions.
    - [x] InceptionRes-v2 layers (or blocks) and their transposed versions.
    - [x] InceptionPlus layers (or blocks) and their transposed versions.
    - [x] External interface for using generic python function.
    - [x] Droupout method options for all avaliable modern layers.
- [ ] data:
    - [x] Basic h5py (HDF5) IO handles.
    - [ ] Basic SQLite IO handles.
    - [ ] Basic Bcolz IO handles.
    - [ ] Basic CSV IO handles.
    - [ ] Basic JSON IO handles.
    - [ ] Data parsing utilities.
- [ ] estimators:
    - [ ] VGG16
    - [ ] U-Net
    - [ ] ResNet
- [x] functions:
    - [x] (loss):    Lovasz loss for IoU
    - [x] (loss):    Linear interpolated loss for IoU
    - [x] (metrics): signal-to-noise ratio (SNR and PSNR)
    - [x] (metrics): Pearson correlation coefficient
    - [x] (metrics): IoU / Jaccard index
- [ ] utilities:
    - [x] Revised save and load model functions.
    - [ ] Beholder plug-in callback.
    - [x] Revised ModelCheckpoint callback.
    - [ ] Extended data visualization tools.

## Demos

Check the branch [`demos`][brch-demos] to learn more details.

## Update records

### 0.60 @ 06/19/2019

1. Support totally new `save_model` and `load_model` APIs in `.utilites`.
2. Finish `ModelCheckpoint` in `.utilities.callbacks`.

### 0.56 @ 06/13/2019

Finish `losses.linear_jaccard_index`, `losses.lovasz_jaccard_loss`, `metrics.signal_to_noise`, `metrics.correlation`, `metrics.jaccard_index` in `.functions` (may require tests in the future).

### 0.54 @ 06/12/2019

1. Add dropout options to all advanced blocks (including residual, ResNeXt, inception, incept-res and incept-plus).
2. Strengthen the compatibility.
3. Fix minor bugs for spatial dropout in `0.50-b`.
4. Thanks to GOD! `.layers` has been finished, although it may require modification in the future.

### 0.50-b @ 06/11/2019

1. Fix a bug for implementing the channel_first mode for `AConv` in `.layers`.
2. Finish `InstanceGaussianNoise` in `.layers`.
3. Prepare the test for adding dropout to residual layers in `.layers`.

### 0.50 @ 06/11/2019

1. Finish `Conv1DTied`, `Conv2DTied`, `Conv3DTied` in `.layers`.
2. Switch back to the 0.48 version for `.layers.DenseTied` APIs because testing show that the modification in 0.48-b will cause bugs.

### 0.48-b @ 06/10/2019

A Test on replacing the `.layers.DenseTied` APIs like `tf.keras.layers.Wrappers`.

### 0.48 @ 06/09/2019

1. Finish `Inceptplus1D`, `Inceptplus2D`, `Inceptplus3D`, `Inceptplus1DTranspose`,  `Inceptplus2DTranspose`, `Inceptplus3DTranspose` in `.layers`.
2. Minor changes for docstrings and default settings in `.layers.inception`.

### 0.45-b @ 06/07/2019

1. Enable the `ResNeXt` to estimate the latent group and local filter number.
2. Make a failed try on implementing quick group convolution, testing results show that using `tf.nn.depthwise_conv2d` to replace multiple `convND` ops would cause the computation to be even slower.

### 0.45 @ 06/06/2019

1. Enable Modern convolutional layers to work with group convolution.
2. Reduce the memory consumption for network construction when using ResNeXt layers in case of out of memory (OOM) problems.
3. Fix a minor bug for group convolution.

### 0.42 @ 06/05/2019

1. Finish `GroupConv1D`, `GroupConv2D`, `GroupConv3D` in `.layers`.
2. Fix the bugs in channel detections for residual and inception layers.

### 0.40 @ 06/05/2019

1. Finish `Resnext1D`, `Resnext2D`, `Resnext3D`, `Resnext1DTranspose`,  `Resnext2DTranspose`, `Resnext3DTranspose` in `.layers`.
2. Fix the repeating biases problems in inception-residual layers.

### 0.38 @ 06/04/2019

1. Finish `Inceptres1D`, `Inceptres2D`, `Inceptres3D`, `Inceptres1DTranspose`,  `Inceptres2DTranspose`, `Inceptres3DTranspose` in `.layers`.
2. Fix some bugs and revise docstrings for `.layers.residual` and `.layers.inception`.

### 0.36 @ 06/01/2019

Finish `Inception1D`, `Inception2D`, `Inception3D`, `Inception1DTranspose`,  `Inception2DTranspose`, `Inception3DTranspose` in `.layers`.

### 0.32 @ 05/31/2019

Finish `Residual1D`, `Residual2D`, `Residual3D`, `Residual1DTranspose`,  `Residual2DTranspose`, `Residual3DTranspose` in `.layers`.

### 0.28 @ 05/24/2019

1. Fix the bug about padding for transposed dilation convolutional layers.
2. Add a new option `output_mshape` to help transposed convolutional layers to control the desired output shape.
3. Finish `PyExternal` in `.layers`.

### 0.24 @ 03/31/2019

Finish `H5GCombiner` in `.data`.

### 0.23 @ 03/27/2019

1. Use `keras.Sequence()` to redefine `H5GParser` and `H5HGParser`.
2. Add compatible check.

### 0.22 @ 03/26/2019

Adjust the `.data.h5py` module to make it more generalized.

### 0.20 @ 03/26/2019

1. Finish `H5HGParser`, `H5SupSaver`, `H5GParser` in `.data`.
2. Finish `DenseTied`, `InstanceNormalization`, `GroupNormalization`, `AConv1D`, `AConv2D`, `AConv3D`, `AConv1DTranspose`,  `AConv2DTranspose`, `AConv3DTranspose` in `.layers`.

### 0.10 @ 03/23/2019

Create this project.

[brch-demos]:https://github.com/cainmagi/MDNT/tree/demos