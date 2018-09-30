import csv
import math
import os

import cv2
import numpy as np
from keras import Model
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras_applications.mobilenet_v2 import _inverted_res_block
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import MaxPooling2D, Conv2D, Reshape, Dense, GlobalAveragePooling2D
from keras.utils import Sequence
from keras.losses import mean_squared_error
import keras.backend as K
from keras.optimizers import Adam
import tensorflow as tf

# 0.35, 0.5, 0.75, 1.0
ALPHA = 1.0

# 96, 128, 160, 192, 224
IMAGE_SIZE = 96

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 50

THREADS = 4

TRAIN_CSV = "../train.csv"
VALIDATION_CSV = "../validation.csv"

CLASSES = 2


class DataSequence(Sequence):

    def __load_images(self, dataset):
        return np.array([cv2.imread(f) for f in dataset], dtype='f')

    def __init__(self, csv_file, batch_size=32, classes=CLASSES, inmemory=False):
        self.paths = []
        self.batch_size = batch_size
        self.inmemory = inmemory

        with open(csv_file, "r") as file:
            self.y = np.zeros((sum(1 for line in file), 4 + classes))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, (path, x0, y0, x1, y1, _, class_id) in enumerate(reader):
                self.y[index][0] = x0
                self.y[index][1] = y0
                self.y[index][2] = x1
                self.y[index][3] = y1
                self.y[index][min(4 + int(class_id), self.y.shape[1]-1)] = 1

                self.paths.append(path)

        if self.inmemory:
            self.x = self.__load_images(self.paths)
            self.x = preprocess_input(self.x)

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.inmemory:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, [batch_y[...,:4], batch_y[...,4:]]

        batch_x = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x)
        images = preprocess_input(images)

        return images, [batch_y[...,:4], batch_y[...,4:]]


def create_model(size, alpha, classes=CLASSES):
    model = MobileNetV2(input_shape=(size, size, 3), include_top=False, alpha=alpha)

    # to freeze layers
    # for layer in model.layers:
    #    layer.trainable = False

    out = model.layers[-1].output

    x = Conv2D(4, kernel_size=3)(out)
    x = Reshape((4,), name="coords")(x)

    y = GlobalAveragePooling2D()(out)
    y = Dense(classes, name="classes", activation="softmax")(y)

    return Model(inputs=model.input, outputs=[x, y])


def iou(y_true, y_pred):
    xA = K.maximum(y_true[...,0], y_pred[...,0])
    yA = K.maximum(y_true[...,1], y_pred[...,1])
    xB = K.minimum(y_true[...,2], y_pred[...,2])
    yB = K.minimum(y_true[...,3], y_pred[...,3])

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (y_true[...,2] - y_true[...,0]) * (y_true[...,3] - y_true[...,1])
    boxBArea = (y_pred[...,2] - y_pred[...,0]) * (y_pred[...,3] - y_pred[...,1])

    return K.clip(interArea / (boxAArea + boxBArea - interArea + K.epsilon()), 0, 1)


def log_mse(y_true, y_pred):
    return K.mean(tf.log1p(tf.squared_difference(y_pred, y_true)), axis=-1)

def focal_loss(alpha=0.9, gamma=2):
  def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
    return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

  def loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    logits = tf.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

  return loss


def train(model, epochs, batch_size, patience, train_csv, validation_csv):
    train_datagen = DataSequence(train_csv, batch_size)
    validation_datagen = DataSequence(validation_csv, batch_size)

    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss={"coords" : log_mse, "classes" : focal_loss()}, loss_weights={"coords" : 1, "classes" : 1}, optimizer=optimizer, metrics={"coords" : iou, "classes" : "accuracy"})
    checkpoint = ModelCheckpoint("model-{val_loss:.2f}.h5", monitor="val_loss", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_loss", patience=patience, mode="auto")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="auto")

    model.summary()

    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        validation_data=validation_datagen,
                        callbacks=[checkpoint, reduce_lr, stop],
                        workers=THREADS,
                        use_multiprocessing=True,
                        shuffle=True,
                        verbose=1)


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    train(model, EPOCHS, BATCH_SIZE, PATIENCE, TRAIN_CSV, VALIDATION_CSV)


if __name__ == "__main__":
    main()
