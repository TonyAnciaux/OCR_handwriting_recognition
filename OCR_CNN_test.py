import numpy as np
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder

width = 640
height = 480
label_encoder = LabelEncoder()


# Function copied from OCR_CNN_model.py to operate the same preprocess to the capture
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    # Easy way to normalize the values (values from 0 to 1)
    img = img / 255
    return img


# Instanciate the "webcam object"
webcam = cv2.VideoCapture(0)
webcam.set(3, width)
webcam.set(4, height)

# Import the model from pickle file ("rb" = Read Bytes)
pickle_in = open("CNN_trained_model_75k", "rb")
model = pickle.load(pickle_in)


while True:
    success, capture = webcam.read()
    image = np.asarray(capture)
    image = cv2.resize(image, (32, 32))
    image = preprocessing(image)
    cv2.imshow("Processed Image", image)
    image = image.reshape(1, 32, 32, 1)

    # prediction / importing the classes created in OCR_CNN_model.py
    label_encoder.classes_ = np.load('classes.npy')
    class_id = np.argmax(model.predict(image), axis=-1)
    class_id = label_encoder.inverse_transform(class_id)

    for label in class_id:
        prediction = []
        if label not in prediction:
            prediction.append(label)
        print(prediction)

    cv2.waitKey(1)
