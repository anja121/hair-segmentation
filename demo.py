import glob
import os
import cv2
import tensorflow.keras as keras
import numpy as np
from utils.train_utils import read_config_file


def process_image(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)

    return im


def color_transfer(image, mask, color, alpha=0.8):
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_n = cv2.merge((mask, mask, mask))

    color_image = np.zeros_like(image)
    color_image[:] = color

    mask_n = mask_n.astype(np.uint8)

    beta = (1.0 - alpha)
    mask_n = cv2.bitwise_and(mask_n, color_image)

    dst = cv2.addWeighted(image, alpha, cv2.GaussianBlur(mask_n, (21, 21), 0), beta, 0.0)

    return dst


def main():
    conf = read_config_file("configs/test_config.json")
    model = keras.models.load_model(conf["model_path"])

    if not os.path.exists(conf["results_folder"]):
        os.makedirs(conf["results_folder"])

    for im_path in glob.glob(conf["path_to_dataset"] + "*.jpg"):
        img = cv2.imread(im_path)
        file_name = os.path.basename(im_path)

        processed_img = process_image(img, height=conf["img_size"], width=conf["img_size"])
        mask = model.predict(processed_img)
        mask = mask.reshape((conf["img_size"], conf["img_size"]))

        b, g, r = conf["color"]
        res_img = color_transfer(img, mask, (b, g, r), alpha=conf["alpha"])

        cv2.imwrite(conf["results_folder"] + "image_" + file_name, img)
        cv2.imwrite(conf["results_folder"] + "result_" + file_name, res_img)


if __name__ == '__main__':
    main()


