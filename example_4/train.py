import csv
import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import epsilon

# 0.35, 0.5, 0.75, 1.0
ALPHA = 1.0

GRID_SIZE = 7
IMAGE_SIZE = 224

# first train with frozen weights, then fine tune
TRAINABLE = False
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 1e-5

MULTITHREADING = True
THREADS = 8

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            lines = sum(1 for line in file)
            file.seek(0)

            self.mask = np.zeros((lines, GRID_SIZE, GRID_SIZE, 5))

            reader = csv.reader(file, delimiter=",")

            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row

                mid_x = 1 / image_width * (x0 + (x1 - x0) / 2)
                mid_y = 1 / image_height * (y0 + (y1 - y0) / 2)

                w = (x1 - x0) / image_width
                h = (y1 - y0) / image_height

                unrounded_x = min(mid_x * GRID_SIZE, GRID_SIZE)
                unrounded_y = min(mid_y * GRID_SIZE, GRID_SIZE)

                cell_x = int(np.floor(unrounded_x))
                cell_y = int(np.floor(unrounded_y))
                cell_w = (x1 - x0) * GRID_SIZE / image_width
                cell_h = (y1 - y0) * GRID_SIZE / image_height

                self.mask[index, :, :, 0] = cell_h
                self.mask[index, :, :, 1] = cell_w
                self.mask[index, :, :, 2] = unrounded_y - cell_y
                self.mask[index, :, :, 3] = unrounded_x - cell_x
                self.mask[index, cell_y - 1, cell_x - 1, 4] = 1

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
        reshaped = mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], -1)

        score_ind = np.argmax(reshaped[...,-1], axis=-1)

        height = reshaped[range(reshaped.shape[0]), score_ind, 0]
        width = reshaped[range(reshaped.shape[0]), score_ind, 1]
        offset_y = reshaped[range(reshaped.shape[0]), score_ind, 2]
        offset_x = reshaped[range(reshaped.shape[0]), score_ind, 3]

        y = score_ind // mask.shape[2]
        x = score_ind % mask.shape[2]

        return np.stack([x + offset_x + 1, y + offset_y + 1, width, height], axis=-1)

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
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    h = tf.clip_by_value(boxes[..., 0], epsilon(), height - epsilon())
    w = tf.clip_by_value(boxes[..., 1], epsilon(), width - epsilon())
    y = tf.clip_by_value(boxes[..., 2], epsilon(), 1.0 - epsilon())
    x = tf.clip_by_value(boxes[..., 3], epsilon(), 1.0 - epsilon())
    s = tf.clip_by_value(boxes[..., 4], epsilon(), 1.0 - epsilon())

    return tf.stack([h, w, y, x, s], axis=-1)

def create_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights="imagenet")

    for layer in model.layers:
        layer.trainable = trainable

    block = model.get_layer("block_16_project_BN").output

    x = Conv2D(320, padding="same", kernel_size=3, strides=1)(block)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(5, padding="same", kernel_size=1, activation=lambda l : tf.concat([l[...,:4], tf.sigmoid(l[...,4:5])], axis=-1))(x)
    x = Lambda(clip)(x)

    return Model(inputs=model.input, outputs=x)

def detection_loss(iou_threshold=0.5):
    def get_box_highest_percentage(arr):
        shape = tf.shape(arr)
        batch, height, width = shape[0], shape[1], shape[2]

        reshaped = tf.reshape(arr, (batch, height * width, -1))

        max_prob_ind = tf.argmax(reshaped[...,-1], axis=1, output_type=tf.int32)
        indices = tf.stack([tf.range(batch), max_prob_ind], axis=-1)

        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        y, x = tf.cast(indices[...,1:2], tf.float32) // height, tf.cast(indices[...,1:2], tf.float32) % width

        out = tf.concat([y, x, tf.gather_nd(reshaped, indices)], axis=-1)
        out = tf.concat([out[...,:1] + out[...,4:5] + 1, out[...,1:2] + out[...,5:6] + 1, out[...,2:4], out[...,6:7]], axis=-1)

        return out

    def calculate_iou(true_box, pred_size):
        shape = tf.shape(pred_size)
        batch, height, width = shape[0], shape[1], shape[2]

        flattened = tf.reshape(tf.range(height * width), (1, -1))
        flattened = tf.tile(flattened, [batch, 1])
        flattened = tf.reshape(flattened, (-1, height * width, 1))

        ys = tf.cast(flattened, tf.float32) // tf.cast(height, tf.float32)
        xs = tf.cast(flattened, tf.float32) % tf.cast(width, tf.float32)
        merged = tf.stack([ys, xs], axis=-1)
        merged = tf.reshape(merged, (batch, height, width, -1))

        pred_boxes = tf.concat([merged[...,:1] + pred_size[...,2:3] + 1, merged[...,1:2] + pred_size[...,3:4] + 1, pred_size[...,:2]], axis=-1)
        pred_boxes = tf.reshape(pred_boxes, (batch, height, width, -1))

        true_boxes = tf.reshape(true_box, (batch, 1, -1))
        true_boxes = tf.tile(true_boxes, [1, height * width, 1])
        true_boxes = tf.reshape(true_boxes, (batch, height, width, -1))

        y0 = tf.maximum(true_boxes[...,0], pred_boxes[...,0])
        x0 = tf.maximum(true_boxes[...,1], pred_boxes[...,1])
        y1 = tf.minimum(true_boxes[...,0] + true_boxes[...,2], pred_boxes[...,0] + pred_boxes[...,2])
        x1 = tf.minimum(true_boxes[...,1] + true_boxes[...,3], pred_boxes[...,1] + pred_boxes[...,3])

        intersection = (x1 - x0) * (y1 - y0)
        union = true_boxes[...,2] * true_boxes[...,3] + pred_boxes[...,2] * pred_boxes[...,3]
        res = tf.clip_by_value(intersection / (union - intersection + epsilon()), epsilon(), 1)
        res = tf.reshape(res, (batch, height, width, 1))

        return res

    def loss(y_true, y_pred):
        obj_true = y_true[...,4:5]
        obj_pred = y_pred[...,4:5]

        true_box = get_box_highest_percentage(y_true)
        pred_box = get_box_highest_percentage(y_pred)

        iou = calculate_iou(true_box, y_pred[...,:-1])
        mask = tf.cast(tf.greater(iou, iou_threshold) | tf.equal(obj_true, 1), tf.float32)

        boxes_true = y_true[...,:-1] * mask
        boxes_pred = y_pred[...,:-1] * mask

        obj_loss = binary_crossentropy(mask, obj_pred)
        size_loss = tf.reduce_mean(tf.squared_difference(boxes_true[...,:2], boxes_pred[...,:2]))
        coord_loss = tf.reduce_mean(tf.squared_difference(boxes_true[...,2:], boxes_pred[...,2:]))

        size_loss2 = tf.reduce_mean(tf.squared_difference(true_box[...,:2], pred_box[...,:2]))
        coord_loss2 = tf.reduce_mean(tf.squared_difference(true_box[...,2:], pred_box[...,2:]))

        return obj_loss + size_loss + coord_loss + size_loss2 + coord_loss2

    return loss

def main():
    model = create_model(trainable=TRAINABLE)
    model.summary()
    #model.load_weights("model-0.55.h5")

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))

    optimizer = SGD(lr=LEARNING_RATE, decay=1e-5, momentum=0.9, nesterov=False)
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