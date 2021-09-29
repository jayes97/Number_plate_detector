from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing._label import LabelEncoder
import numpy as np
import cv2

app = Flask(__name__)

json_file = open('models\MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("models\License_character_recognition.h5")

# text_pred_model = load_model('models/model_004.h5')
labels = LabelEncoder()
labels.classes_ = np.load('models/license_character_classes.npy')

#XML file to detect Region of Interest
plate_cascade = cv2.CascadeClassifier('models/indian_license_plate.xml')


#detecting number plate in image
def detect_plate(img, text=''):
    plate_img_with_car = img.copy()
    roi = img.copy()  # Regioin Of interest
    plate_rect = plate_cascade.detectMultiScale(plate_img_with_car, scaleFactor=1.2, minNeighbors=7)

    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y + h, x:x + w, :]
        plate = roi[y:y + h, x:x + w, :]
        cv2.rectangle(plate_img_with_car, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)

    if text != '':
        plate_img_with_car = cv2.putText(plate_img_with_car, text, (x - w // 2, y - h // 2),
                                         cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)

    return plate

#finding contours of each number
def find_contours(dimensions, plate_img):
    cntrs, _ = cv2.findContours(plate_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # 15 largest counters
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    plate_img_for_cntrs = plate_img.copy()

    x_cntr_list = []
    # target_contours = []
    char_img_list = []

    for cntr in cntrs:

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:

            x_cntr_list.append(intX)
            char_copy = np.zeros((44, 24))

            char = plate_img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(plate_img_for_cntrs, (intX, intY), (intWidth + intX, intHeight + intY), (50, 21, 200), 2)

            # reverting colors
            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            char_img_list.append(char_copy)

   #sorting characters as per sequence
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])

    char_img_list_sorted = []
    for idx in indices:
        char_img_list_sorted.append(char_img_list[idx])

    char_img_list = np.array(char_img_list_sorted)

    return char_img_list

#image preprocessing before finding contours
def segment_characters(plate_image):
    img_lp = cv2.resize(plate_image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH, LP_HEIGHT = img_binary_lp.shape[:2]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]
    cv2.imwrite('contour.jpg', img_binary_lp)

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

#predicting from pre-trained model
def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction



@app.route('/', methods=['GET'])
def helloWorld():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():

    imagefile = request.files['imagefile']
    image_path = "images/" + imagefile.filename
    imagefile.save(image_path)
    img = cv2.imread(image_path)
    plate = detect_plate(img)
    
    char_list = segment_characters(plate)
    
    final_string =''
    
    for i, character in enumerate(char_list):

        char = np.array2string(predict_from_model(character,model,labels))
        
        final_string+= char.strip("'[]")

    print(final_string)
    return render_template("index.html", prediction=final_string, img_source=image_path)
    # return "hello world"


if __name__== "__main__":
    app.run(host='0.0.0.0', port=8080)
