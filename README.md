# object-localization
This project shows how to localize a single object in an image by just using a convolutional neural network. There are more sophisticated methods like YOLO, R-CNN, SSD or RetinaNet (focal loss), but sometimes the only thing you need are the coordinates of a single object.

# Architecture
To avoid pretraining the CNN and having the highest speed, the MobileNet [1] model was chosen as a base. I did all the training with alpha = 0.25 and an image size of 128x128. It is also possible to use higher values for a better accuracy (check out [2]).

In order to train for detection, the dense layer was removed and instead I added another depthwise separable convolution block, followed by max pooling and a normal convolution for the coordinates.

If you want to also distinguish between different objects, the loss function has to be adjusted. The following function should work (not tested):

```
def custom_loss(y_true, y_pred):
    box_loss = losses.mean_squared_error(y_true[...,:4], y_pred[...,:4])
    obj_loss = losses.binary_crossentropy(y_true[...,4:], y_pred[...,4:])

    return box_loss + obj_loss
```

Then add another conv layer with softmax, followed by a merge layer. Furthermore, if you want to improve the performance, the MSE can be replaced also by a smooth L1 loss. Square rooting the weight/height is a possibility to improve accuracy for smaller objects (with some weighting maybe). See the YOLO/SSD paper for more information.

# Installation
1. Download "The Oxford-IIIT Pet Dataset" [3].
2. Put the groundtruth data and the dataset in one folder.
3. Run generate_dataset.py
4. Run train_model.py
5. Run evaluate_performance.py

# Result
I stopped training after 5 hours, but the IoU was still improving. Using a CPU for training just took too long.

In the following images red is the predicted box, green is the ground truth:

![Image 1](https://i.imgur.com/pArUlGd.jpg)

![Image 2](https://i.imgur.com/ll9PNOF.jpg)

# References

[1] A. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications.

[2] https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

[3] http://www.robots.ox.ac.uk/~vgg/data/pets/
