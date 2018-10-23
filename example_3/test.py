from train_model import *
import cv2
import glob

WEIGHTS_FILE = "model-0.27.h5"
IMAGES = "../images/*jpg"

def main():
    model = create_model()
    model.load_weights(WEIGHTS_FILE)

    for filename in glob.glob(IMAGES):
        unscaled = cv2.imread(filename)
        image = cv2.resize(unscaled, (IMAGE_WIDTH, IMAGE_HEIGHT))
        feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

        region = model.predict(x=np.array([feat_scaled]))[0]

        boxes = []
        for j in range(region.shape[1]):
            for i in range(region.shape[0]):
                if region[i][j] > 0.5:
                    x = int(CELL_WIDTH * j * unscaled.shape[1] / IMAGE_WIDTH)
                    y = int(CELL_HEIGHT * i * unscaled.shape[0] / IMAGE_HEIGHT)
                    x2 = int(CELL_WIDTH * (j + 1) * unscaled.shape[1] / IMAGE_WIDTH)
                    y2 = int(CELL_HEIGHT * (i + 1) * unscaled.shape[0] / IMAGE_HEIGHT)
                    #cv2.rectangle(unscaled, (x, y), (x2, y2), (0, 255, 0), 2)
                    if not boxes or boxes[-1][2] < x:
                        boxes.append([x, y, x2, y2])
                    else:
                        boxes[-1][0] = min(x, boxes[-1][0])
                        boxes[-1][1] = min(y, boxes[-1][1])
                        boxes[-1][2] = max(x2, boxes[-1][2])
                        boxes[-1][3] = max(y2, boxes[-1][3])

        for box in boxes:
            cv2.rectangle(unscaled, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

        cv2.imshow("image", unscaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
