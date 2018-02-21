import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from generate_dataset import DATASET_FOLDER
from train_model import create_model, IMAGE_SIZE, ALPHA

DEBUG = False
WEIGHTS_FILE = "model-0.81.h5"


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = (xB - xA) * (yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_ = interArea / (boxAArea + boxBArea - interArea)

    return max(iou_, 0)


def main():
    model = create_model(IMAGE_SIZE, ALPHA)
    model.load_weights(WEIGHTS_FILE)

    ious = []
    xmls = sorted(glob.glob("{}/*xml".format(DATASET_FOLDER)))
    for i, xml_file in enumerate(xmls):
        print("{}/{}".format(i + 1, len(xmls)), end="\r")
        tree = ET.parse(xml_file)

        path = os.path.join(DATASET_FOLDER, tree.findtext("./filename"))
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        xmin = int(tree.findtext("./object/bndbox/xmin"))
        ymin = int(tree.findtext("./object/bndbox/ymin"))
        xmax = int(tree.findtext("./object/bndbox/xmax"))
        ymax = int(tree.findtext("./object/bndbox/ymax"))

        # hardcoded fix, the box is wrong
        if "Abyssinian_1.jpg" in path:
            xmin -= 160
            xmax -= 160

        box1 = [(xmin / width) * IMAGE_SIZE, (ymin / height) * IMAGE_SIZE, (xmax / width) * IMAGE_SIZE,
                (ymax / height) * IMAGE_SIZE]

        image = cv2.resize(cv2.imread(path), (IMAGE_SIZE, IMAGE_SIZE))
        region = model.predict(x=np.array([image]))[0]
        x, y, w, h = region

        xmin = x - w / 2
        ymin = y - h / 2
        xmax = xmin + w
        ymax = ymin + h

        box2 = [xmin, ymin, xmax, ymax]

        iou_ = iou(box1, box2)
        ious.append(iou_)

        if DEBUG:
            print("IoU for {} is {}".format(path, iou_))
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
            cv2.rectangle(image, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 255, 0), 1)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    np.set_printoptions(suppress=True)
    print("\nAvg IoU: {}".format(np.mean(ious)))
    print("Highest IoU: {}".format(np.max(ious)))
    print("Lowest IoU: {}".format(np.min(ious)))


if __name__ == "__main__":
    main()
