import csv
import math

import cv2
import numpy as np
from keras import Model
from keras.applications.mobilenetv2 import MobileNetV2
from keras_applications.mobilenet_v2 import _inverted_res_block
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import MaxPooling2D, Conv2D, Reshape
from keras.utils import Sequence

# 0.35, 0.5, 0.75, 1.0
ALPHA = 0.35

# 96, 128, 160, 192, 224
IMAGE_SIZE = 96

EPOCHS = 10000
BATCH_SIZE = 32
PATIENCE = 500

MEAN = np.array([[[336.45663766, 336.45663766, 336.45663766]]])

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"


class DataSequence(Sequence):

    def __load_images(self, dataset):
        return np.array([cv2.imread(f) for f in dataset], dtype='f')

    def __init__(self, csv_file, batch_size, mean, inmemory=False):
        self.paths = []
        self.mean = mean
        self.batch_size = batch_size
        self.inmemory = inmemory

        with open(csv_file, "r") as file:
            self.y = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, (path, x0, y0, x1, y1) in enumerate(reader):
                self.y[index][0] = x0
                self.y[index][1] = y0
                self.y[index][2] = x1
                self.y[index][3] = y1

                self.paths.append(path)

        if self.inmemory:
            self.x = self.__load_images(self.paths)
            self.x -= self.mean

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.inmemory:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

        batch_x = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x)
        images -= self.mean

        return images, batch_y


def create_model(size, alpha):
    model = MobileNetV2(input_shape=(size, size, 3), include_top=False, alpha=alpha)

    # to freeze layers
    # for layer in model.layers:
    #     layer.trainable = False

    x = model.layers[-1].output

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=17)

    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(4, kernel_size=(1, 1), padding="same")(x)
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)


def train(model, epochs, batch_size, patience, train_csv, validation_csv, mean):
    train_datagen = DataSequence(train_csv, batch_size, mean)
    validation_datagen = DataSequence(validation_csv, batch_size, mean)

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    checkpoint = ModelCheckpoint("model-{val_loss:.2f}.h5", monitor="val_loss", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_loss", patience=patience, mode="auto")

    model.summary()

    model.fit_generator(train_datagen, epochs=epochs, validation_data=validation_datagen,
                        callbacks=[checkpoint, stop])


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    train(model, EPOCHS, BATCH_SIZE, PATIENCE, TRAIN_CSV, VALIDATION_CSV, MEAN)


if __name__ == "__main__":
    main()
