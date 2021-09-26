import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing  import LabelEncoder
import numpy as np
import cv2
from image_processing import detect_plate, segment_characters
from model import predict_from_model

app = Flask(__name__)

text_pred_model = load_model('models/model_004.h5')
labels = LabelEncoder()
labels.classes_ = np.load('models/license_character_classes.npy')

@app.route("/predict", methods=['GET'])
def predict():

    # imagefile = request.files['imagefile']
    # image_path = "images/" + imagefile.filename
    # imagefile.save(image_path)
    # img = cv2.imread(image_path)
    # plate = detect_plate(img)
    #
    # char_list = segment_characters(plate)
    #
    # final_string =''
    #
    # for i, character in enumerate(char_list):
    #     char = predict_from_model(character,text_pred_model,labels)
    #     final_string+= char.strip("'[]")
    #
    # return render_template("index.html", prediction=final_string)
    return "hello world"


if __name__=="__main__":
    app.run(debug=True)