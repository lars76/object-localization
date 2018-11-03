from train import *
import cv2
import glob

WEIGHTS_FILE = "model-0.34.h5"
IMAGES = "images/*jpg"
THRESHOLD = 0.5
EPSILON = 0.02

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        unscaled = cv2.imread(filename)
        image = cv2.resize(unscaled, (IMAGE_WIDTH, IMAGE_HEIGHT))
        feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

        region = model.predict(x=np.array([feat_scaled]))[0]

        output = np.zeros(unscaled.shape[:2], dtype=np.uint8)
        for i in range(region.shape[1]):
            for j in range(region.shape[0]):
                if region[i][j] > THRESHOLD:
                    x = int(CELL_WIDTH * j * unscaled.shape[1] / IMAGE_WIDTH)
                    y = int(CELL_HEIGHT * i * unscaled.shape[0] / IMAGE_HEIGHT)
                    x2 = int(CELL_WIDTH * (j + 1) * unscaled.shape[1] / IMAGE_WIDTH)
                    y2 = int(CELL_HEIGHT * (i + 1) * unscaled.shape[0] / IMAGE_HEIGHT)
                    #cv2.rectangle(unscaled, (x, y), (x2, y2), (0, 255, 0), 1)

                    output[y:y2,x:x2] = 1

        _, contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, EPSILON * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(unscaled, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
