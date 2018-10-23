import csv
import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Reshape, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


# 0.35, 0.5, 0.75, 1.0
ALPHA = 1.0

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

HEIGHT_CELLS = 8
WIDTH_CELLS = 8

CELL_WIDTH = IMAGE_WIDTH / WIDTH_CELLS
CELL_HEIGHT = IMAGE_HEIGHT / HEIGHT_CELLS

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 50

THREADS = 4

TRAIN_CSV = "../train.csv"
VALIDATION_CSV = "../validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            lines = sum(1 for line in file)
            file.seek(0)

            self.y = np.zeros((lines - 1, HEIGHT_CELLS, WIDTH_CELLS))

            reader = csv.reader(file, delimiter=",")
            next(reader)

            for index, row in enumerate(reader):
                filename = row[0]
                x, y, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])

                xmin = x - w / 2
                ymin = y - h / 2

                xmax = xmin + w
                ymax = ymin + h

                cell_start_x = max(math.ceil(xmin / CELL_WIDTH) - 1, 0)
                cell_stop_x = min(math.ceil(xmax / CELL_WIDTH) - 1, WIDTH_CELLS - 1)

                cell_start_y = max(math.ceil(ymin / CELL_HEIGHT) - 1, 0)
                cell_stop_y = min(math.ceil(ymax / CELL_HEIGHT) - 1, HEIGHT_CELLS - 1)

                for k in range(cell_start_y, cell_stop_y+1):
                    for k2 in range(cell_start_x, cell_stop_x+1):
                        self.y[index, k, k2] = 1

                self.paths.append(filename)

    def __len__(self):
        return math.ceil(len(self.y) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_x = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_y = self.y[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        x = np.zeros((len(batch_x), IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        for i, f in enumerate(batch_x):
            img = Image.open(f)
            x[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return x, batch_y

def create_model():
    model = MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, alpha=ALPHA, weights="imagenet")
    out = model.layers[-1].output

    x = GlobalAveragePooling2D()(out)
    x = Dense(HEIGHT_CELLS * WIDTH_CELLS, activation="sigmoid")(x)
    x = Reshape((HEIGHT_CELLS, WIDTH_CELLS))(x)

    return Model(inputs=model.input, outputs=x)

def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return numerator / (denominator + tf.keras.backend.epsilon())

def loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - tf.log(dice_coefficient(y_true, y_pred) + tf.keras.backend.epsilon())

def main():
    model = create_model()
    model.summary()

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = DataGenerator(VALIDATION_CSV)

    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer, metrics=[dice_coefficient])
    
    checkpoint = ModelCheckpoint("model-{val_loss:.2f}.h5", monitor="val_loss", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="auto")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="auto")

    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        validation_data=validation_datagen,
                        callbacks=[checkpoint, reduce_lr, stop],
                        workers=THREADS,
                        use_multiprocessing=True,
                        shuffle=True,
                        verbose=1)


if __name__ == "__main__":
    main()
