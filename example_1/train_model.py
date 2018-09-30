import csv
import math

import cv2
import numpy as np
from keras import Model
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras_applications.mobilenet_v2 import _inverted_res_block
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import MaxPooling2D, Conv2D, Reshape
from keras.utils import Sequence
import keras.backend as K

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


class DataSequence(Sequence):

    def __load_images(self, dataset):
        return np.array([cv2.imread(f) for f in dataset], dtype='f')

    def __init__(self, csv_file, batch_size=32, inmemory=False):
        self.paths = []
        self.batch_size = batch_size
        self.inmemory = inmemory

        with open(csv_file, "r") as file:
            self.y = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, (path, x0, y0, x1, y1, _, _) in enumerate(reader):
                self.y[index][0] = x0
                self.y[index][1] = y0
                self.y[index][2] = x1
                self.y[index][3] = y1

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
            return batch_x, batch_y

        batch_x = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x)
        images = preprocess_input(images)

        return images, batch_y

def create_model(size, alpha):
    model = MobileNetV2(input_shape=(size, size, 3), include_top=False, alpha=alpha)

    # to freeze layers
    # for layer in model.layers:
    #     layer.trainable = False

    x = model.layers[-1].output
    x = Conv2D(4, kernel_size=3, name="coords")(x)
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)


def iou(y_true, y_pred):
    xA = K.maximum(y_true[...,0], y_pred[...,0])
    yA = K.maximum(y_true[...,1], y_pred[...,1])
    xB = K.minimum(y_true[...,2], y_pred[...,2])
    yB = K.minimum(y_true[...,3], y_pred[...,3])

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (y_true[...,2] - y_true[...,0]) * (y_true[...,3] - y_true[...,1])
    boxBArea = (y_pred[...,2] - y_pred[...,0]) * (y_pred[...,3] - y_pred[...,1])

    return K.clip(interArea / (boxAArea + boxBArea - interArea + K.epsilon()), 0, 1)


def train(model, epochs, batch_size, patience, train_csv, validation_csv):
    train_datagen = DataSequence(train_csv, batch_size)
    validation_datagen = DataSequence(validation_csv, batch_size)

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[iou])
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
