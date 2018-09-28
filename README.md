# object-localization

This project shows how to localize a single object in an image by just using a convolutional neural network. There are more sophisticated methods like YOLO, R-CNN, SSD or RetinaNet (focal loss), but sometimes the only thing you need are the coordinates of a single object.

# Architecture

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

The code in this repository uses MobileNetv2 [1], because it is faster than other models and the performance can be adapted. For example, if alpha = 0.35 with 96x96 is not good enough, one can just increase both values (see [2] for a comparison). If you use another architecture, change `preprocess_input`.

# Example: Finding cats and dogs in images

## Installation

1. pip3 install imgaug (needed for data augmentations)
2. Download [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
3. Download [The Oxford-IIIT Pet Dataset Annotations](http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz)
4. tar xf images.tar.gz
5. tar xf annotations.tar.gz
6. mv annotations/xmls/* images/
7. python3 generate_dataset.py
8. python3 train_model.py
9. Adjust the WEIGHTS_FILE in evaluate_performance.py (given by the last script)
10. python3 evaluate_performance.py

## Result

I trained the neural network for 75 epochs. The results are 90% avg IoU on training set, 72% on validation set.

Configuration: no augmentations, full fine tuning (no freezing), image size 96, alpha 1.0, batch size 32, initial learning rate 0.001 (then decrease).

In the following images red is the predicted box, green is the ground truth:

![Image 1](https://i.imgur.com/pArUlGd.jpg)

![Image 2](https://i.imgur.com/ll9PNOF.jpg)

# Guidelines

## Improve accuracy (IoU)

- enable augmentations: set `AUGMENTATION=True` in generate_dataset.py and install *imgaug*.
- better augmentations: increase `AUGMENTATION_PER_IMAGE` and try out different transformations.
- for MobileNetv1/2: increase `ALPHA` and `IMAGE_SIZE` in train_model.py
- other architectures: increase `IMAGE_SIZE`
- add more layers: e.g. YOLOv2 adds 3 conv layers
- read https://github.com/keras-team/keras/pull/9965

## Increase training speed

- set `inmemory=True` in train_model.py for small datasets
- increase `BATCH_SIZE`
- less layers, `IMAGE_SIZE` and `ALPHA`

## Overfitting

- If the new dataset is small and similar to ImageNet, freeze all layers.
- If the new dataset is small and not similar to ImageNet, freeze some layers.
- If the new dataset is large, freeze no layers.
- read http://cs231n.github.io/transfer-learning/

# References

[1] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.

[2] https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
