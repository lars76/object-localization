import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import csv

from generate_dataset import TRAIN_OUTPUT_FILE, VALIDATION_OUTPUT_FILE, DATASET_FOLDER
from train_model import create_model, IMAGE_SIZE, ALPHA, MEAN

DEBUG = False
WEIGHTS_FILE = "model-43.63.h5"

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_ = interArea / (boxAArea + boxBArea - interArea)
    if iou_ > 1:
        iou_ = 0

    return max(iou_, 0)

def convert_coords(x, y, w, h):
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = xmin + w
    ymax = ymin + h

    return np.floor(np.array([xmin, ymin, xmax, ymax])).astype(int)

def predict_image(path, model):
    im = cv2.imread(path)
    if im.shape[0] != IMAGE_SIZE:
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))

    image = np.array(im, dtype='f')
    image -= MEAN

    region = model.predict(x=np.array([image]))[0]

    return convert_coords(*region)

def show_image(path, ground_truth, pred):
    image = cv2.imread(path)

    if ground_truth.any():
        cv2.rectangle(image, (ground_truth[0], ground_truth[1]), (ground_truth[2], ground_truth[3]), (0, 255, 0), 1)
    if pred.any():
        cv2.rectangle(image, (pred[0], pred[1]), (pred[2], pred[3]), (0, 0, 255), 1)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dataset_iou(csv_file, model):
    with open(csv_file, "r") as f:
        lines = sum(1 for row in f)
        f.seek(0)

        ious = []
        reader = csv.reader(f, delimiter=",")
        for i, (path, xmin, ymin, xmax, ymax) in enumerate(reader):
            print("{}/{}".format(i + 1, lines), end="\r")

            coords = (float(xmin), float(ymin), float(xmax), float(ymax))
            ground_truth = convert_coords(*coords)

            pred = predict_image(path, model)

            iou_ = iou(ground_truth, pred)
            ious.append(iou_)

            if DEBUG:
                print("IoU for {} is {}".format(path, iou_))
                show_image(path, ground_truth, pred)

        np.set_printoptions(suppress=True)
        print("\nAvg IoU: {}".format(np.mean(ious)))
        print("Highest IoU: {}".format(np.max(ious)))
        print("Lowest IoU: {}".format(np.min(ious)))

def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    model.load_weights(WEIGHTS_FILE)

    print("IoU on training data")
    dataset_iou(TRAIN_OUTPUT_FILE, model)

    print("\nIoU on validation data")
    dataset_iou(VALIDATION_OUTPUT_FILE, model)

    print("\nTrying out unscaled image")
    for k in os.listdir(DATASET_FOLDER):
        if "jpg" in k:
            break
    path = os.path.join(DATASET_FOLDER, k)
    pred = predict_image(path, model)

    height, width, _ = cv2.imread(path).shape
    scaled = np.array([pred[0] * width / IMAGE_SIZE, pred[1] * height / IMAGE_SIZE,
                      pred[2] * width / IMAGE_SIZE, pred[3] * height / IMAGE_SIZE])
    show_image(path, np.array([0]), np.floor(scaled).astype(int))

    print("\nDone")


if __name__ == "__main__":
    main()
