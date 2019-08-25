from train import *
import cv2
import glob
import numpy as np

WEIGHTS_FILE = "model-0.76.h5"
IMAGES = "images/*jpg"

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2
MAX_OUTPUT_SIZE = 49

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        unscaled = cv2.imread(filename)
        img = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))

        feat_scaled = preprocess_input(np.array(img, dtype=np.float32))

        pred = np.squeeze(model.predict(feat_scaled[np.newaxis,:]))
        height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]

        coords = np.arange(pred.shape[0] * pred.shape[1])
        y = (y_f + coords // pred.shape[0]) / pred.shape[0]
        x = (x_f + coords % pred.shape[1]) / pred.shape[1]

        boxes = np.stack([y, x, height, width, score], axis=-1)
        boxes = boxes[np.where(boxes[...,-1] >= SCORE_THRESHOLD)]

        selected_indices = tf.image.non_max_suppression(boxes[...,:-1], boxes[...,-1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
        selected_indices = tf.Session().run(selected_indices)

        for y_c, x_c, h, w, _ in boxes[selected_indices]:
            x0 = unscaled.shape[1] * (x_c - w / 2)
            y0 = unscaled.shape[0] * (y_c - h / 2)
            x1 = x0 + unscaled.shape[1] * w
            y1 = y0 + unscaled.shape[0] * h

            cv2.rectangle(unscaled, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)

        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()