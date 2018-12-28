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

AUGMENTATION = False
AUGMENTATION_DEBUG = False
AUGMENTATION_PER_IMAGE = 25
AUGMENTATION_FOLDER = "augmentations"

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except ImportError:
    print("Augmentation disabled")
    AUGMENTATION = False


def generate_images(row):
    path, height, width, xmin, ymin, xmax, ymax, class_name, class_id = row

    image = cv2.imread(path)

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Affine(scale=(0.4, 1.6)),
        iaa.Crop(percent=(0, 0.2)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.AddToHueAndSaturation((-30, 30)),
        iaa.Sometimes(0.5,
                      iaa.Affine(rotate=(-45, 45)),
                      iaa.Affine(shear=(-16, 16))),
        iaa.Sometimes(0.2,
                      iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                         children=iaa.WithChannels(0, iaa.Add((10, 50)))),
                      ),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.GaussianBlur(sigma=(0, 1.0))
    ])

    new_rows = []
    for i in range(0, AUGMENTATION_PER_IMAGE):
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        if not bbs_aug.bounding_boxes[0].is_fully_within_image(image.shape):
            continue
        # Another possibility is:
        # bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()
        # if not bbs_aug.bounding_boxes:
        #    continue
        after = bbs_aug.bounding_boxes[0]

        if AUGMENTATION_DEBUG:
            image_aug = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

        name, ftype = os.path.splitext(os.path.basename(path))
        new_filename = "{}_aug_{}{}".format(name, i, ftype)
        if not os.path.exists(AUGMENTATION_FOLDER):
            os.makedirs(AUGMENTATION_FOLDER)

        new_path = os.path.abspath(os.path.join(AUGMENTATION_FOLDER, new_filename))
        cv2.imwrite(new_path, image_aug)

        new_rows.append([new_path, height, width, after.x1, after.y1, after.x2, after.y2, class_name, class_id])

    return new_rows

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

        # hardcoded fix, the box is wrong
        #if "Abyssinian_1.jpg" in path:
        #    xmin -= 160
        #    xmax -= 160

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
                if AUGMENTATION and i <= c * SPLIT_RATIO:
                    aug = generate_images([path, width, height, xmin, ymin, xmax, ymax, class_name, class_names[class_name]])
                    for k in aug:
                        writer.writerow(k)

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
