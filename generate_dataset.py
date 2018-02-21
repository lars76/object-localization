import csv
import glob
import os
import re
import xml.etree.ElementTree as ET

DATASET_FOLDER = "dataset/"
TRAIN_OUTPUT_FILE = "train.csv"
VALIDATION_OUTPUT_FILE = "validation.csv"
DICTIONARY_OUTPUT_FILE = "dictionary.txt"

SPLIT_RATIO = 0.8
MAX_CLASSES = 37

AUGMENTATION = True
AUGMENTATION_DEBUG = False
AUGMENTATION_PER_IMAGE = 25
AUGMENTATION_FOLDER = "augmentation/"

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
    import cv2
except ImportError:
    print("Augmentation disabled")
    AUGMENTATION = False


def generate_images(row):
    path, class_id, width, height, xmin, ymin, xmax, ymax = row
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
                      iaa.Affine(shear=(-16, 16))
                      ),
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

        new_filename = "{}_aug_{}".format(i, os.path.basename(path))
        new_path = os.path.join(AUGMENTATION_FOLDER, new_filename)
        cv2.imwrite(new_path, image_aug)

        new_rows.append([new_path, class_id, width, height, after.x1, after.y1, after.x2, after.y2])

    return new_rows


def main():
    if not os.path.exists(DATASET_FOLDER):
        print("Data set not found")
        return
    if not os.path.exists(AUGMENTATION_FOLDER) and AUGMENTATION:
        os.makedirs(AUGMENTATION_FOLDER)

    with open(TRAIN_OUTPUT_FILE, "w") as train, open(VALIDATION_OUTPUT_FILE, "w") as validate, open(
            DICTIONARY_OUTPUT_FILE, "w") as dictionary:
        writer = csv.writer(train, delimiter=",")
        writer2 = csv.writer(validate, delimiter=",")

        already_seen = []
        class_id = 0
        animal = []
        for xml_file in sorted(glob.glob("{}/*xml".format(DATASET_FOLDER))):
            tree = ET.parse(xml_file)

            animal_type = os.path.basename(xml_file).replace(".xml", "")
            animal_type = re.sub(r"[^a-zA-Z]+", " ", animal_type).strip().lower()
            name = "{} ({})".format(animal_type, tree.findtext("./object/name"))
            if not name in already_seen:
                print("{}/{}".format(class_id, MAX_CLASSES), end="\r")
                already_seen.append(name)
                for index, a in enumerate(animal):
                    if index <= len(animal) * SPLIT_RATIO:
                        if AUGMENTATION:
                            aug = generate_images(a)
                            for k in aug:
                                writer.writerow(k)
                        writer.writerow(a)
                    else:
                        writer2.writerow(a)
                if class_id == MAX_CLASSES:
                    break
                dictionary.write("{}\n".format(name))
                class_id += 1
                animal = []

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

            row = [path, class_id, width, height, xmin, ymin, xmax, ymax]
            animal.append(row)

    print("Done!")


if __name__ == "__main__":
    main()
