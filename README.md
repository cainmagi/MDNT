# Modern Deep Network Toolkits for Tensorflow-Keras

We proudly present our newest produce, a totally well-defined extension for Tensorflow-Keras users!

## Instruction

### First download

For the users who need to clone this branch first, please use such commands to make sure that the submodule would be loaded correctly:

```bash
git clone -b demos https://github.com/cainmagi/MDNT.git mdnt-demos
cd mdnt-demos
git submodule update --init --recursive
```

### Update

To make the demo scripts updated to the newest version, please use such a command in the project folder:

```bash
git pull
```

To make the MDNT library inside this branch updated to the newest version, please use such a command in the project folder:

```bash
git submodule update --recursive --remote
```

## Demos

And we have already written these demos:

The following demos are working for denoising with MNIST data set.

* `demo-AEdense`: Tied autoencoder based on dense layers.
* `demo-normalize`: Normalization layers.
* `demo-AEConvTied`: Tied autoencoder based on trivial convolutional layers.
* `demo-AEConv1DTied`: Tied autoencoder based on trivial 1D convolutional layers.
* `demo-GroupConv`: Plain group convolutional layers (without normalization).
* `demo-AConv`: Modern convolutional layers.
* `demo-AConvTranspose`: Autoencoder based on modern convolutional layers.
* `demo-AConv1DTranspose`: Autoencoder based on modern 1D convolutional layers.
* `demo-AGroupConvTranspose`: Autoencoder based on modern group convolutional layers.
* `demo-ResTranspose`: Autoencoder based on modern residual-v2 layers.
* `demo-ResNeXtTranspose`: Autoencoder based on modern ResNeXt (residual-v3) layers.
* `demo-InceptTranspose`: Autoencoder based on modern inception-v4 layers.
* `demo-InceptResTranspose`: Autoencoder based on modern inception-residual-v2 layers.
* `demo-InceptPlusTranspose`: Autoencoder based on modern inception-plus layers.
* `demo-external-1`: External layer test 1. We use numpy-api to convert Cartesian coordinates to polar coordinates.
* `demo-saveH5`: Save `.h5` file.
* `demo-readH5`: Read `.h5` file.
* `demo-readH5Multi`: Read two `.h5` files and combine them as a data set.
* `demo-dataSet`: Use H5 IO to train and test a network.

## Update records

### @ 06/12/2019

1. Enable `demo-InceptPlusTranspose` to work on the test for dropout methods.

### @ 06/11/2019

1. Update the tests (`demo-AEConvTied`, `demo-AEConv1DTied`) for tied convolutional (`Conv2DTied`, `Conv1DTied`) layers.
2. Add tensorboard logger for `demo-AEdense`.
3. Enable `demo-ResTranspose` to work on the test for dropout methods.

### @ 06/09/2019

1. Update the test (`demo-InceptPlusTranspose`) for inception plus (`Inceptplus`) layers.
2. Minor revisions on docstrings.
3. Make the learning rate schedulers for some tests optional.

### @ 06/07/2019

1. Solve the memory collision problem in `demo-ResNeXtTranspose` for ResNeXt (`Resnext`) layers.
2. Update the test (`demo-AGroupConvTranspose`) for modern group convolutional (`AConv` with `lgroups`) layers.

### @ 06/05/2019

1. Update the test (`demo-ResNeXtTranspose`) for ResNeXt (`Resnext`) layers. (Current codes have memory collision.)
2. Update the test (`demo-GroupConv`) for group convolutional (`GroupConv`) layers.

### @ 06/04/2019

1. Update the test (`demo-InceptResTranspose`) for inception-residual (`Inceptres`) layers.
2. Revise the docstring for `demo-ResTranspose`.

### @ 06/01/2019

1. Move all demos into the branch `demos`.
2. Add the master branch as the submodule.