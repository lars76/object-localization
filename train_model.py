import csv
import math

import cv2
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence

ALPHA = 0.25
IMAGE_SIZE = 128

EPOCHS = 5000
PATIENCE = 100


class DataSequence(Sequence):

    def __load_images(self, dataset):
        out = []
        for file_name in dataset:
            im = cv2.resize(cv2.imread(file_name), (self.image_size, self.image_size))
            out.append(im)

        return np.array(out)

    def __init__(self, csv_file, image_size, batch_size=32, feature_scaling=False):
        self.csv_file = csv_file
        with open(self.csv_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            arr = list(reader)

        self.y = np.zeros((len(arr), 4))
        self.x = []
        self.image_size = image_size

        for index, (path, class_id, width, height, x0, y0, x1, y1) in enumerate(arr):
            width, height, x0, y0, x1, y1 = int(width), int(height), int(x0), int(y0), int(x1), int(y1)
            mid_x = x0 + (x1 - x0) / 2
            mid_y = y0 + (y1 - y0) / 2
            self.y[index][0] = (mid_x / width) * IMAGE_SIZE
            self.y[index][1] = (mid_y / height) * IMAGE_SIZE
            self.y[index][2] = ((x1 - x0) / width) * IMAGE_SIZE
            self.y[index][3] = ((y1 - y0) / height) * IMAGE_SIZE
            self.x.append(path)

        self.batch_size = batch_size
        self.feature_scaling = feature_scaling
        if self.feature_scaling:
            dataset = self.__load_images(self.x)
            broadcast_shape = [1, 1, 1]
            broadcast_shape[2] = dataset.shape[3]

            self.mean = np.mean(dataset, axis=(0, 1, 2))
            self.mean = np.reshape(self.mean, broadcast_shape)
            self.std = np.std(dataset, axis=(0, 1, 2))
            self.std = np.reshape(self.std, broadcast_shape) + K.epsilon()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x).astype('float32')
        if self.feature_scaling:
            images -= self.mean
            images /= self.std

        return images, batch_y


def create_model(size, alpha):
    model_net = MobileNet(input_shape=(size, size, 3), include_top=False, alpha=alpha)
    x = _depthwise_conv_block(model_net.layers[-1].output, 1024, alpha, 1, block_id=14)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(4, kernel_size=(1, 1), padding="same")(x)
    x = Reshape((4,))(x)

    return Model(inputs=model_net.input, outputs=x)


def train(model, epochs, image_size):
    train_datagen = DataSequence("train.csv", image_size)
    validation_datagen = DataSequence("validation.csv", image_size)

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    checkpoint = ModelCheckpoint("model-{val_acc:.2f}.h5", monitor="val_acc", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="auto")

    model.fit_generator(train_datagen, steps_per_epoch=1150, epochs=epochs, validation_data=validation_datagen,
                        validation_steps=22, callbacks=[checkpoint, stop])


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    train(model, EPOCHS, IMAGE_SIZE)


if __name__ == "__main__":
    main()
