# object-localization

This project shows how to localize objects in images by using simple convolutional neural networks.

# Dataset

Before getting started, we have to download a dataset and generate a csv file containing the annotations (boxes).

1. Download [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
2. Download [The Oxford-IIIT Pet Dataset Annotations](http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz)
3. tar xf images.tar.gz
4. tar xf annotations.tar.gz
5. mv annotations/xmls/* images/
6. python3 generate_dataset.py

# Single-object detection

## Example 1: Finding dogs/cats

### Architecture

First, let's look at YOLOv2's approach:

1. Pretrain Darknet-19 on ImageNet (feature extractor)
2. Remove the last convolutional layer
3. Add three 3 x 3 convolutional layers with 1024 filters
4. Add a 1 x 1 convolutional layer with the number of outputs needed for detection

We proceed in the same way to build the object detector:

1. Choose a model from [Keras Applications](https://keras.io/applications/) i.e. feature extractor
2. Remove the dense layer
3. Freeze some/all/no layers
3. Add one/multiple/no convolution block (or `_inverted_res_block` for MobileNetv2)
4. Add a convolution layer for the coordinates

The code in this repository uses MobileNetv2, because it is faster than other models and the performance can be adapted. For example, if alpha = 0.35 with 96x96 is not good enough, one can just increase both values (see [here](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py) for a comparison). If you use another architecture, change `preprocess_input`.

1. `python3 example_1/train.py`
2. Adjust the WEIGHTS_FILE in `example_1/test.py` (given by the last script)
3. `python3 example_1/test.py`

### Result

In the following images red is the predicted box, green is the ground truth:

![Image 1](https://i.imgur.com/pArUlGd.jpg)

![Image 2](https://i.imgur.com/ll9PNOF.jpg)

## Example 2: Finding dogs/cats and distinguishing classes

This time we have to run the scripts `example_2/train.py` and `example_2/test.py`.

### Changes

In order to distinguish between classes, we have to modify the loss function. I'm using here `w_1*log((y_hat - y)^2 + 1) + w_2*FL(p_hat, p)` where `w_1 = w_2 = 1` are two weights and `FL(p_hat, p) = -(0.9(1 - p_hat)^2 p*log(p_hat) + 0.1*p_hat^2(1 - p)log(1-p_hat))` (focal loss). 

Instead of using all 37 classes, the code will only output class 0 (contains only class 0) or class 1 (contains class 1 to 36). However, it is easy to extend this to more classes (use categorical cross entropy instead of focal loss and try out different weights).

# Multi-object detection

## Example 3: Segmentation-like detection

### Architecture

In this example, we use a skip-net architecture similar to U-Net. For an in-depth explanation see [my blog post](https://lars76.github.io/neural-networks/object-detection/obj-detection-using-segmentation/).

![Architecture](https://lars76.github.io/assets/images/architecture.png)

### Result

![Dog](https://lars76.github.io/assets/images/dog2.gif)

## Example 4: YOLO-like detection

### Architecture

This example is based on the three YOLO papers. For an in-depth explanation see [this blog post](https://lars76.github.io/neural-networks/object-detection/obj-detection-from-scratch/).

### Result

![Multiple dogs](https://lars76.github.io/assets/images/multiple_dogs.jpg)

# Guidelines

## Improve accuracy (IoU)

- enable augmentations: see `example_4` the same code can be added to the other examples
- better augmentations: try out different values (flips, rotation etc.)
- for MobileNetv1/2: increase `ALPHA` and `IMAGE_SIZE` in train_model.py
- other architectures: increase `IMAGE_SIZE`
- add more layers
- try out other loss functions (MAE, smooth L1 loss etc.)
- other optimizer: SGD with momentum 0.9, adjust learning rate
- use a feature pyramid
- read https://github.com/keras-team/keras/pull/9965

## Increase training speed

- increase `BATCH_SIZE`
- less layers, `IMAGE_SIZE` and `ALPHA`

## Overfitting

- If the new dataset is small and similar to ImageNet, freeze all layers.
- If the new dataset is small and not similar to ImageNet, freeze some layers.
- If the new dataset is large, freeze no layers.
- read http://cs231n.github.io/transfer-learning/
