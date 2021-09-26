import cv2
import numpy as np


def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction