import csv
import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D, Reshape, BatchNormalization, Activation
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import epsilon

# 0.35, 0.5, 0.75, 1.0
ALPHA = 1.0

GRID_SIZE = 28
IMAGE_SIZE = 224

# first train with frozen weights, then fine tune
TRAINABLE = False
WEIGHTS = "model-0.89.h5"

EPOCHS = 200
BATCH_SIZE = 8
PATIENCE = 15

MULTI_PROCESSING = False
THREADS = 1

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.mask = np.zeros((sum(1 for line in file), GRID_SIZE, GRID_SIZE))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")

            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row

                cell_start_x = np.rint(((GRID_SIZE - 1) / image_width) * x0).astype(int)
                cell_stop_x = np.rint(((GRID_SIZE - 1) / image_width) * x1).astype(int)

                cell_start_y = np.rint(((GRID_SIZE - 1) / image_height) * y0).astype(int)
                cell_stop_y = np.rint(((GRID_SIZE - 1) / image_height) * y1).astype(int)

                self.mask[index, cell_start_y : cell_stop_y, cell_start_x : cell_stop_x] = 1

                self.paths.append(path)

    def __len__(self):
        return math.ceil(len(self.mask) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_masks = self.mask[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img.convert('RGB')

            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return batch_images, batch_masks[:,:,:,np.newaxis]

class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        numerator = 0
        denominator = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            numerator += 2 * np.sum(gt * pred)
            denominator += np.sum(gt + pred)

        dice = np.round(numerator / denominator, 4)
        logs["val_dice"] = dice

        print(" - val_dice: {}".format(dice))

def create_model(trainable=True):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights="imagenet")

    for layer in model.layers:
        layer.trainable = trainable

    block1 = model.get_layer("block_5_add").output
    block2 = model.get_layer("block_12_add").output
    block3 = model.get_layer("block_15_add").output

    blocks = [block2, block1]

    x = block3
    for block in blocks:
        x = UpSampling2D()(x)

        x = Conv2D(256, kernel_size=3, padding="same", strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Concatenate()([x, block])

        x = Conv2D(256, kernel_size=3, padding="same", strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)

    return Model(inputs=model.input, outputs=x)

def loss(y_true, y_pred):
    def dice_coefficient(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

        return numerator / (denominator + epsilon())

    return binary_crossentropy(y_true, y_pred) - tf.math.log(dice_coefficient(y_true, y_pred) + epsilon())

def main():
    model = create_model(trainable=TRAINABLE)
    model.summary()

    if TRAINABLE:
        model.load_weights(WEIGHTS)

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))

    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer, metrics=[])
    
    checkpoint = ModelCheckpoint("model-{val_dice:.2f}.h5", monitor="val_dice", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max")
    stop = EarlyStopping(monitor="val_dice", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_dice", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="max")

    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                        workers=THREADS,
                        use_multiprocessing=MULTI_PROCESSING,
                        shuffle=True,
                        verbose=1)


if __name__ == "__main__":
    main()