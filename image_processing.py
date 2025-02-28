import cv2
import numpy as np

def prepare_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img
