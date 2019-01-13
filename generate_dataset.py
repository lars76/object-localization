import csv
import cv2
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

DATASET_FOLDER = "images/"
TRAIN_OUTPUT_FILE = "train.csv"
VALIDATION_OUTPUT_FILE = "validation.csv"

SPLIT_RATIO = 0.8

def main():
    if not os.path.exists(DATASET_FOLDER):
        print("Dataset not found")
        return

    class_names = {}
    k = 0
    output = []
    xml_files = glob.glob("{}/*xml".format(DATASET_FOLDER))
    for i, xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)

        path = os.path.join(DATASET_FOLDER, tree.findtext("./filename"))

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        xmin = int(tree.findtext("./object/bndbox/xmin"))
        ymin = int(tree.findtext("./object/bndbox/ymin"))
        xmax = int(tree.findtext("./object/bndbox/xmax"))
        ymax = int(tree.findtext("./object/bndbox/ymax"))

        basename = os.path.basename(path)
        basename = os.path.splitext(basename)[0]
        class_name = basename[:basename.rfind("_")].lower()
        if class_name not in class_names:
            class_names[class_name] = k
            k += 1

        output.append((path, height, width, xmin, ymin, xmax, ymax, class_name, class_names[class_name]))

    # preserve percentage of samples for each class ("stratified")
    output.sort(key=lambda tup : tup[-1])

    lengths = []
    i = 0
    last = 0
    for j, row in enumerate(output):
        if last == row[-1]:
            i += 1
        else:
            print("class {}: {} images".format(output[j-1][-2], i))
            lengths.append(i)
            i = 1
            last += 1

    print("class {}: {} images".format(output[j-1][-2], i))
    lengths.append(i)

    with open(TRAIN_OUTPUT_FILE, "w") as train, open(VALIDATION_OUTPUT_FILE, "w") as validate:
        writer = csv.writer(train, delimiter=",")
        writer2 = csv.writer(validate, delimiter=",")

        s = 0
        for c in lengths:
            for i in range(c):
                print("{}/{}".format(s + 1, sum(lengths)), end="\r")

                path, height, width, xmin, ymin, xmax, ymax, class_name, class_id = output[s]

                if xmin >= xmax or ymin >= ymax or xmax > width or ymax > height or xmin < 0 or ymin < 0:
                    print("Warning: {} contains invalid box. Skipped...".format(path))
                    continue

                row = [path, height, width, xmin, ymin, xmax, ymax, class_name, class_names[class_name]]
                if i <= c * SPLIT_RATIO:
                    writer.writerow(row)
                else:
                    writer2.writerow(row)

                s += 1

    print("\nDone!")

    """ preprocess_input is as good as exact mean/std
    print("Calculating mean and std...")

    mean = 0
    std = 0
    length = 0
    images = glob.glob("{}/*".format(TRAIN_FOLDER))
    for i, path in enumerate(images):
        print("{}/{}".format(i + 1, len(images)), end="\r")
        sum_ = np.mean(cv2.imread(path))

        length += 1

        mean_next = mean + (sum_ - mean) / length
        std += (sum_ - mean) * (sum_ - mean_next)
        mean = mean_next

    std = np.sqrt(std / (length - 1))

    print("\nMean: {}".format(mean))
    print("Std: {}".format(std))
    """


if __name__ == "__main__":
    main()
