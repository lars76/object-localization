from train import *
import cv2
import glob

WEIGHTS_FILE = "model-0.91.h5"
IMAGES = "images/*jpg"
THRESHOLD = 0.5
EPSILON = 0.02

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        unscaled = cv2.imread(filename)
        image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
        feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

        region = np.squeeze(model.predict(feat_scaled[np.newaxis,:]))

        output = np.zeros(region.shape, dtype=np.uint8)
        output[region > 0.5] = 1

        contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, EPSILON * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            x0 = np.rint(x * unscaled.shape[1] / output.shape[1]).astype(int)
            x1 = np.rint((x + w) * unscaled.shape[1] / output.shape[1]).astype(int)
            y0 = np.rint(y * unscaled.shape[0] / output.shape[0]).astype(int)
            y1 = np.rint((y + h) * unscaled.shape[0] / output.shape[0]).astype(int)
            cv2.rectangle(unscaled, (x0, y0), (x1, y1), (0, 255, 0), 1)

        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
