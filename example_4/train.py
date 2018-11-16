import csv
import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import epsilon


# 0.35, 0.5, 0.75, 1.0
ALPHA = 0.35

GRID_SIZE = 14
IMAGE_SIZE = 224

TRAINABLE = False
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 1e-3

MULTITHREADING = True
THREADS = 4

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            lines = sum(1 for line in file)
            file.seek(0)

            self.mask = np.zeros((lines, GRID_SIZE, GRID_SIZE, 3))

            reader = csv.reader(file, delimiter=",")

            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row

                mid_x = 1 / image_width * (x0 + (x1 - x0) / 2)
                mid_y = 1 / image_height * (y0 + (y1 - y0) / 2)

                w = (x1 - x0) / image_width
                h = (y1 - y0) / image_height

                cell_x = int(min(np.rint(mid_x * GRID_SIZE), GRID_SIZE)) - 1
                cell_y = int(min(np.rint(mid_y * GRID_SIZE), GRID_SIZE)) - 1
                cell_w = w * GRID_SIZE
                cell_h = h * GRID_SIZE

                self.mask[index, :, :, 0] = cell_h
                self.mask[index, :, :, 1] = cell_w
                self.mask[index, cell_y, cell_x, 2] = 1

                self.paths.append(path)
 

    def __len__(self):
        return math.ceil(len(self.paths) / BATCH_SIZE)

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

        return batch_images, batch_masks


class Validation(Callback):
    def get_box(self, mask):
        reshaped = mask.reshape(mask.shape[0], -1, 3)

        score_ind = np.argmax(reshaped[...,2], axis=-1)

        height = reshaped[range(reshaped.shape[0]), score_ind, 0]
        width = reshaped[range(reshaped.shape[0]), score_ind, 1]

        y = score_ind // mask.shape[2]
        x = score_ind % mask.shape[2]

        return np.stack([x, y, width, height], axis=-1)

    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)

            pred = self.get_box(pred)
            gt = self.get_box(gt)

            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)
            gt = np.maximum(gt, 0)

            diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
            diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:,2] * gt[:,3]
            area_pred = pred[:,2] * pred[:,3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        print(" - val_iou: {} - val_mse: {}".format(iou, mse))

def clip(boxes):
    _, height, width, _ = boxes.shape

    h = tf.clip_by_value(boxes[..., 0], 0.0, tf.cast(height, tf.float32))
    w = tf.clip_by_value(boxes[..., 1], 0.0, tf.cast(width, tf.float32))
    s = tf.clip_by_value(boxes[..., 2], epsilon(), 1.0)

    return tf.stack([h, w, s], axis=-1)

def create_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights="imagenet")

    for layer in model.layers:
        layer.trainable = trainable

    #block0 = model.get_layer("Conv1_relu").output
    #block1 = model.get_layer("block_2_add").output
    #block2 = model.get_layer("block_5_add").output
    block3 = model.get_layer("block_12_add").output
    block4 = model.get_layer("block_15_add").output

    blocks = [block3]#, block2, block1, block0]

    x = block4
    for block in blocks:
        x = UpSampling2D()(x)
        x = Concatenate()([x, block])

    x = Conv2D(256, kernel_size=3, padding="same", strides=1, activation="relu")(x)
    x = Conv2D(256, kernel_size=3, padding="same", strides=1, activation="relu")(x)
    x = Conv2D(3, kernel_size=1, activation=lambda l : tf.concat([l[...,:2], tf.sigmoid(l[...,2:])], axis=-1))(x)

    x = Lambda(clip)(x)

    return Model(inputs=model.input, outputs=x)

def detection_loss(alpha=0.9, gamma=2, threshold=0.5):
    def obj_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon(), 1 - epsilon())
        logits = tf.log(y_pred / (1 - y_pred))

        weight_a = alpha * (1 - y_pred) ** gamma * y_true
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - y_true)

        loss_ = (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

        return tf.reduce_sum(loss_)

    def coord_loss(y_true, y_pred, prob):
        diff = y_true - y_pred
        loss_ = tf.where(tf.less(diff, 1.0), 0.5 * diff ** 2, tf.abs(diff) - 0.5)

        return tf.log1p(tf.reduce_sum(prob ** gamma * loss_))

    def loss(y_true, y_pred):
        obj_true = y_true[...,2:]
        obj_pred = y_pred[...,2:]

        coords_true = y_true[...,:2] * tf.cast(tf.greater(obj_pred, threshold), tf.float32)
        coords_pred = y_pred[...,:2] * tf.cast(tf.greater(obj_pred, threshold), tf.float32)

        return obj_loss(obj_true, obj_pred) + coord_loss(coords_true, coords_pred, obj_pred)

    return loss

def main():
    model = create_model(trainable=TRAINABLE)
    model.summary()

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))

    optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=detection_loss(), optimizer=optimizer, metrics=[])
    
    checkpoint = ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max", period=1)
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=5, min_lr=1e-6, verbose=1, mode="max")

    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                        workers=THREADS,
                        use_multiprocessing=MULTITHREADING,
                        shuffle=True,
                        verbose=1)


if __name__ == "__main__":
    main()
