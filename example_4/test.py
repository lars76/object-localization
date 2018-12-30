from train import *
import cv2
import glob
import numpy as np

WEIGHTS_FILE = "model-0.37.h5"
IMAGES = "images/*jpg"

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
MAX_OUTPUT_SIZE = 300

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        unscaled = cv2.imread(filename)
        image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
        feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

        pred = model.predict(x=np.array([feat_scaled]))[0]
        height, width, y, x, score = pred[..., 0].flatten(), pred[..., 1].flatten(), pred[..., 2].flatten(), pred[..., 3].flatten(), pred[..., 4].flatten()

        coords = np.arange(pred.shape[0] * pred.shape[1])
        boxes = np.stack([coords // pred.shape[0] + y + 1, coords % pred.shape[1] + x + 1, height, width, score], axis=-1)
        boxes = boxes[np.where(boxes[...,-1] >= SCORE_THRESHOLD)]

        selected_indices = tf.image.non_max_suppression(boxes[...,:-1], boxes[...,-1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
        selected_indices = tf.Session().run(selected_indices)

        for k in boxes[selected_indices]:
            h = k[2] * unscaled.shape[0]
            w = k[3] * unscaled.shape[1]

            y0 = k[0] * unscaled.shape[0] / pred.shape[0] - h / 2
            x0 = k[1] * unscaled.shape[1] / pred.shape[1] - w / 2
            y1 = y0 + h
            x1 = x0 + w

            #cv2.rectangle(unscaled, (int(k[1] * unscaled.shape[0] / pred.shape[0]), int(k[0] * unscaled.shape[0] / pred.shape[0])), (int(10 + k[1] * unscaled.shape[0] / pred.shape[0]), int(10 + k[0] * unscaled.shape[0] / pred.shape[0])), (0, 0, 255), 1)
            cv2.rectangle(unscaled, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)

        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
