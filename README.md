# Number_plate_detector
Detects numbers of number plate

# Data Gathering
All Data is gathered from Kaggle
1. A pretrained xml file to detect Region of Interest (number plate) is gathered frome [here](https://www.kaggle.com/sarthakvajpayee/ai-indian-license-plate-recognition-data)
2. Train and validation Image data is taken from [here](https://www.kaggle.com/nainikagaur/dataset-characters)

**Model is trained using MobileNet V2**
You can find model training ipynb file in training/training.ipynb directory

### app.py File contains the code run the file on local host using **Flask**

### ../models/ directory contains the saved pre-trained model weights as License_character_recognition.h5
### and model itself as MobileNets_character_recognition.json
