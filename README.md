# Modern Deep Network Toolkits for Tensorflow-Keras

We proudly present our newest produce, a totally well-defined extension for Tensorflow-Keras users!

## Documentation

Still not available now, will implement in the future.

## Progress

Now we have such progress on the semi-product:

- [x] optimzers:
    - [x] Manually switched optimizers (`Adam2SGD` and `NAdam2NSGD`).
    - [x] Automatically switched optimizer (`SWATS`).
    - [x] Advanced adaptive optimizers ( `Adabound`, `Nadabound` and `MNadam` supporting `amsgrad`).
    - [x] Wrapped default optimizers.
- [x] layers:
    - [x] Ghost layer (used to construct trainable input layer).
    - [x] Tied dense layer for the symmetric autoencoder.
    - [x] Extended dropout and noise layers.
    - [x] Extended activation layers.
    - [x] Extended normalization layers.
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
    - [x] LossWeightsScheduler callback (for changing the loss weights during the training).
    - [x] OptimizerSwitcher callback (for using manually switched optimizers).
    - [x] ModelWeightsReducer callback (parameter decay strategy including L1 decay and L2 decay).
    - [ ] Extended data visualization tools.

## Demos

Check the branch [`demos`][brch-demos] to learn more details.

## Update records

### 0.73 @ 10/24/2019

1. Fix a bug for `H5GCombiner` in `.data` when adding more parsers.
2. Finish `H5VGParser` in `.data`, this parser is used for splitting validation set from a dataset.
3. Finish `ExpandDims` in `.layers`, it is a layer version of `tf.expand_dims`.
4. Enable `ModelCheckpoint` in `.utilities.callbacks` to support the option for not saving optimizer.

### 0.72 @ 10/22/2019

1. Fix a bug for serializing `Ghost` in `.layers`.
2. Finish activation layers in `.layers`, including `Slice`, `Restrict` and `RestrictSub`.

### 0.70 @ 10/15/2019

1. Let `.save_model`/`.load_model` supports storing/recovering variable loss weights.
2. Finish `LossWeightsScheduler` in `.utilities.callbacks`.

### 0.69-b @ 10/07/2019

Enable the `H5SupSaver` in `.data` to add more data to an existed file.
    
### 0.69 @ 09/10/2019

Enable the `H5SupSaver` in `.data` to expand if data is dumped in series.

### 0.68 @ 06/27/2019

1. Finish `MNadam`, `Adabound` and `Nadabound` in `.optimizers`.
2. Slightly change `.optimizers.mixture`.
3. Change the quick interface in `.optimizers`.

### 0.64-b @ 06/26/2019

1. Finish the demo version for `SWATS` in `.optimizers`. Need further tests.
2. Fix a small bug for `.load_model`.
3. Change the warning backend to tensorflow version.

### 0.64 @ 06/24/2019

1. Finish `ModelWeightsReducer` in `.utilities.callbacks`.
2. Finish `Ghost` in `.layers`.
3. Fix small bugs.

### 0.63 @ 06/23/2019

1. Fix the bugs of manually switched optimizers in `.optimizers.` Now they require to be used with a callback or switch the phase by `switch()`.
2. Add a plain momentum SGD optimizer to fast interface in `.optimizers`.
3. Finish `OptimizerSwitcher` in `.utilities.callbacks`. It is used to control the phase of the manually swtiched optimizers.
4. Improve the efficiency for `Adam2SGD` and `NAdam2NSGD` in `.optimizers`.

### 0.62 @ 06/21/2019

1. Finish the manually switched optimizers in `.optimizers`: `Adam2SGD` and `NAdam2NSGD`. Both of them supports amsgrad mode.
2. Adjust the fast interface `.optimizers.optimizer`. Now it supports 2 more tensorflow based optimizers and the default momentum of Nesterov SGD optimizer is changed to 0.9.

### 0.60-b @ 06/20/2019

1. Fix some bugs in `.layers.conv` and `.layers.unit`.
2. Remove the normalization layer from all projection branches in `.layers.residual` and `.layers.inception`.

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