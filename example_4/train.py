import csv
import math
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageEnhance
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import epsilon
from tensorflow.keras.models import model_from_json

# 0.35, 0.5, 0.75, 1.0
ALPHA = 0.35

GRID_SIZE = 7
IMAGE_SIZE = 224

# first train with frozen weights, then fine tune
TRAINABLE = False
WEIGHTS = "model-0.64.h5"

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0005
LR_DECAY = 0.0001

MULTITHREADING = False
THREADS = 1

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file, rnd_rescale=True, rnd_multiply=True, rnd_color=True, rnd_crop=True, rnd_flip=False, debug=False):
        self.boxes = []
        self.rnd_rescale = rnd_rescale
        self.rnd_multiply = rnd_multiply
        self.rnd_color = rnd_color
        self.rnd_crop = rnd_crop
        self.rnd_flip = rnd_flip
        self.debug = debug

        with open(csv_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row
                self.boxes.append((path, x0, y0, x1, y1))

    def __len__(self):
        return math.ceil(len(self.boxes) / BATCH_SIZE)

    def __getitem__(self, idx):
        boxes = self.boxes[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(boxes), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(boxes), GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
        for i, row in enumerate(boxes):
            path, x0, y0, x1, y1 = row

            with Image.open(path) as img:
                if self.rnd_rescale:
                    old_width = img.width
                    old_height = img.height

                    rescale = np.random.uniform(low=0.6, high=1.4)
                    new_width = int(old_width * rescale)
                    new_height = int(old_height * rescale)

                    img = img.resize((new_width, new_height))

                    x0 *= new_width / old_width
                    y0 *= new_height / old_height
                    x1 *= new_width / old_width
                    y1 *= new_height / old_height

                if self.rnd_crop:
                    start_x = np.random.randint(0, high=np.floor(0.15 * img.width))
                    stop_x = img.width - np.random.randint(0, high=np.floor(0.15 * img.width))
                    start_y = np.random.randint(0, high=np.floor(0.15 * img.height))
                    stop_y = img.height - np.random.randint(0, high=np.floor(0.15 * img.height))

                    img = img.crop((start_x, start_y, stop_x, stop_y))

                    x0 = max(x0 - start_x, 0)
                    y0 = max(y0 - start_y, 0)
                    x1 = min(x1 - start_x, img.width)
                    y1 = min(y1 - start_y, img.height)

                    if np.abs(x1 - x0) < 5 or np.abs(y1 - y0) < 5:
                        print("\nWarning: cropped too much (obj width {}, obj height {}, img width {}, img height {})\n".format(x1 - x0, y1 - y0, img.width, img.height))

                if self.rnd_flip:
                    elem = np.random.choice([0, 90, 180, 270, 1423, 1234])
                    if elem % 10 == 0:
                        x = x0 - img.width / 2
                        y = y0 - img.height / 2

                        x0 = img.width / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
                        y0 = img.height / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

                        x = x1 - img.width / 2
                        y = y1 - img.height / 2

                        x1 = img.width / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
                        y1 = img.height / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

                        img = img.rotate(-elem)
                    else:
                        if elem == 1423:
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            y0 = img.height - y0
                            y1 = img.height - y1

                        elif elem == 1234:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            x0 = img.width - x0
                            x1 = img.width - x1

                image_width = img.width
                image_height = img.height

                tmp = x0
                x0 = min(x0, x1)
                x1 = max(tmp, x1)

                tmp = y0
                y0 = min(y0, y1)
                y1 = max(tmp, y1)

                x0 = max(x0, 0)
                y0 = max(y0, 0)

                y0 = min(y0, image_height)
                x0 = min(x0, image_width)
                y1 = min(y1, image_height)
                x1 = min(x1, image_width)

                if self.rnd_color:
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(np.random.uniform(low=0.5, high=1.5))

                    enhancer2 = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(np.random.uniform(low=0.7, high=1.3))

                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32)

                if self.rnd_multiply:
                    img[...,0] = np.floor(np.clip(img[...,0] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))
                    img[...,1] = np.floor(np.clip(img[...,1] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))
                    img[...,2] = np.floor(np.clip(img[...,2] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))

                batch_images[i] = preprocess_input(img.copy())

            x_c = (GRID_SIZE / image_width) * (x0 + (x1 - x0) / 2)
            y_c = (GRID_SIZE / image_height) * (y0 + (y1 - y0) / 2)

            floor_y = math.floor(y_c)
            floor_x = math.floor(x_c)

            batch_boxes[i, floor_y, floor_x, 0] = (y1 - y0) / image_height
            batch_boxes[i, floor_y, floor_x, 1] = (x1 - x0) / image_width
            batch_boxes[i, floor_y, floor_x, 2] = y_c - floor_y
            batch_boxes[i, floor_y, floor_x, 3] = x_c - floor_x
            batch_boxes[i, floor_y, floor_x, 4] = 1

            if self.debug:
                changed = img.astype(np.uint8)
                if not os.path.exists("__debug__"):
                    os.makedirs("__debug__")

                changed = Image.fromarray(changed)

                x_c = (floor_x + batch_boxes[i, floor_y, floor_x, 3]) / GRID_SIZE
                y_c = (floor_y + batch_boxes[i, floor_y, floor_x, 2]) / GRID_SIZE

                y0 = IMAGE_SIZE * (y_c - batch_boxes[i, floor_y, floor_x, 0] / 2)
                x0 = IMAGE_SIZE * (x_c - batch_boxes[i, floor_y, floor_x, 1] / 2)
                y1 = y0 + IMAGE_SIZE * batch_boxes[i, floor_y, floor_x, 0]
                x1 = x0 + IMAGE_SIZE * batch_boxes[i, floor_y, floor_x, 1]

                draw = ImageDraw.Draw(changed)
                draw.rectangle(((x0, y0), (x1, y1)), outline="green")

                changed.save(os.path.join("__debug__", os.path.basename(path)))

        return batch_images, batch_boxes


class Validation(Callback):
    def get_box_highest_percentage(self, mask):
        reshaped = mask.reshape(mask.shape[0], np.prod(mask.shape[1:-1]), -1)

        score_ind = np.argmax(reshaped[...,-1], axis=-1)
        unraveled = np.array(np.unravel_index(score_ind, mask.shape[:-1])).T[:,1:]

        cell_y, cell_x = unraveled[...,0], unraveled[...,1]
        boxes = mask[np.arange(mask.shape[0]), cell_y, cell_x]

        h, w, offset_y, offset_x = boxes[...,0], boxes[...,1], boxes[...,2], boxes[...,3]

        return np.stack([cell_y + offset_y, cell_x + offset_x,
                        GRID_SIZE * h, GRID_SIZE * w], axis=-1)

    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)  # in tf2.0: add .numpy()

            pred = self.get_box_highest_percentage(pred)
            gt = self.get_box_highest_percentage(gt)

            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)
            gt = np.maximum(gt, 0)

            diff_height = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
            diff_width = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
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

def create_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights="imagenet")

    for layer in model.layers:
        layer.trainable = trainable

    block = model.get_layer("block_16_project_BN").output

    x = Conv2D(112, padding="same", kernel_size=3, strides=1, activation="relu")(block)
    x = Conv2D(112, padding="same", kernel_size=3, strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(5, padding="same", kernel_size=1, activation="sigmoid")(x)

    model = Model(inputs=model.input, outputs=x)

    # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
    # see https://arxiv.org/pdf/1711.05101.pdf
    regularizer = l2(WEIGHT_DECAY / 2)
    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope("weight_regularizer"):
            model.add_loss(regularizer(weight)) # in tf2.0: lambda: regularizer(weight)

    return model

def detection_loss():
    def get_box_highest_percentage(arr):
        shape = tf.shape(arr)

        reshaped = tf.reshape(arr, (shape[0], tf.reduce_prod(shape[1:-1]), -1))

        # returns array containing the index of the highest percentage of each batch
        # where 0 <= index <= height * width
        max_prob_ind = tf.argmax(reshaped[...,-1], axis=-1, output_type=tf.int32)

        # turn indices (batch, y * x) into (batch, y, x)
        # returns (3, batch) tensor
        unraveled = tf.unravel_index(max_prob_ind, shape[:-1])

        # turn tensor into (batch, 3) and keep only (y, x)
        unraveled = tf.transpose(unraveled)[:,1:]
        y, x = unraveled[...,0], unraveled[...,1]

        # stack indices and create (batch, 5) tensor which
        # contains height, width, offset_y, offset_x, percentage
        indices = tf.stack([tf.range(shape[0]), y, x], axis=-1)
        box = tf.gather_nd(arr, indices)

        y, x = tf.cast(y, tf.float32), tf.cast(x, tf.float32)

        # transform box to (y + offset_y, x + offset_x, 7 * height, 7 * width, obj)
        # output is (batch, 5)
        out = tf.stack([y + box[...,2], x + box[...,3],
                        GRID_SIZE * box[...,0], GRID_SIZE * box[...,1],
                        box[...,-1]], axis=-1)

        return out

    def loss(y_true, y_pred):
        # get the box with the highest percentage in each image
        true_box = get_box_highest_percentage(y_true)
        pred_box = get_box_highest_percentage(y_pred)

        # object loss
        obj_loss = binary_crossentropy(y_true[...,4:5], y_pred[...,4:5])

        # mse with the boxes that have the highest percentage
        box_loss = tf.reduce_sum(tf.math.squared_difference(true_box[...,:-1], pred_box[...,:-1]))

        return tf.reduce_sum(obj_loss) + box_loss

    return loss

def main():
    model = create_model(trainable=TRAINABLE)
    model.summary()

    if TRAINABLE:
        model.load_weights(WEIGHTS)

    train_datagen = DataGenerator(TRAIN_CSV)

    val_generator = DataGenerator(VALIDATION_CSV, rnd_rescale=False, rnd_multiply=False, rnd_crop=False, rnd_flip=False, debug=False)
    validation_datagen = Validation(generator=val_generator)

    learning_rate = LEARNING_RATE
    if TRAINABLE:
        learning_rate /= 10

    optimizer = SGD(lr=learning_rate, decay=LR_DECAY, momentum=0.9, nesterov=False)
    model.compile(loss=detection_loss(), optimizer=optimizer, metrics=[])

    checkpoint = ModelCheckpoint("model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max")
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.6, patience=5, min_lr=1e-6, verbose=1, mode="max")

    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                        workers=THREADS,
                        use_multiprocessing=MULTITHREADING,
                        shuffle=True,
                        verbose=1)


if __name__ == "__main__":
    main()