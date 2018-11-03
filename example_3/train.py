import csv
import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D, Reshape
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


# 0.35, 0.5, 0.75, 1.0
ALPHA = 1.0

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

HEIGHT_CELLS = 28
WIDTH_CELLS = 28

CELL_WIDTH = IMAGE_WIDTH / WIDTH_CELLS
CELL_HEIGHT = IMAGE_HEIGHT / HEIGHT_CELLS

EPOCHS = 200
BATCH_SIZE = 8
PATIENCE = 15

THREADS = 4

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.mask = np.zeros((sum(1 for line in file), HEIGHT_CELLS, WIDTH_CELLS))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")

            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row

                x0 *= IMAGE_WIDTH / image_width
                y0 *= IMAGE_HEIGHT / image_height
                x1 *= IMAGE_WIDTH / image_width
                y1 *= IMAGE_HEIGHT / image_height 

                cell_start_x = max(math.ceil(x0 / CELL_WIDTH) - 1, 0)
                cell_stop_x = min(math.ceil(x1 / CELL_WIDTH) - 1, WIDTH_CELLS - 1)

                cell_start_y = max(math.ceil(y0 / CELL_HEIGHT) - 1, 0)
                cell_stop_y = min(math.ceil(y1 / CELL_HEIGHT) - 1, HEIGHT_CELLS - 1)

                self.mask[index, cell_start_y:cell_stop_y+1, cell_start_x:cell_stop_x+1] = 1

                self.paths.append(path)

    def __len__(self):
        return math.ceil(len(self.mask) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_masks = self.mask[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            img = img.convert('RGB')

            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return batch_images, batch_masks

def create_model(trainable=True):
    model = MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, alpha=ALPHA, weights="imagenet")

    for layer in model.layers:
        layer.trainable = trainable

    block1 = model.get_layer("block_5_add").output
    block2 = model.get_layer("block_12_add").output
    block3 = model.get_layer("block_15_add").output

    x = Concatenate()([UpSampling2D()(block3), block2])
    x = Concatenate()([UpSampling2D()(x), block1])

    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)
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

    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
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
