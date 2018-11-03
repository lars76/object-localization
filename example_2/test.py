import glob
import cv2
import numpy as np

from train import create_model, IMAGE_SIZE
from keras.applications.mobilenetv2 import preprocess_input

WEIGHTS_FILE = "model-0.29.h5"
IMAGES = "images/*jpg"

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        unscaled = cv2.imread(filename)
        image_height, image_width, _ = unscaled.shape

        image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
        feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

        region, class_id = model.predict(x=np.array([image]))
        region = region[0]

        x0 = int(region[0] * image_width / IMAGE_SIZE)
        y0 = int(region[1]  * image_height / IMAGE_SIZE)

        x1 = int((region[0] + region[2]) * image_width / IMAGE_SIZE)
        y1 = int((region[1] + region[3]) * image_height / IMAGE_SIZE)

        class_id = np.argmax(class_id, axis=1)

        cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(unscaled, "class: {}".format(class_id[0]), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
