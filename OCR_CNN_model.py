"""
Built using The Chars74K & NIST Special Database 19
"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import pickle

path = "char_data/"
image_dimension = (32, 32, 3)


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    # Normalize the values (values from 0 to 1)
    img = img / 255
    return img


def my_model():
    no_filters = 60
    size_filter_1 = (5, 5)
    size_filter_2 = (3, 3)
    size_pool = (2, 2)
    no_nodes = 500
    model = Sequential()
    model.add((Conv2D(no_filters, size_filter_1,
                      input_shape=(image_dimension[0], image_dimension[1], 1),
                      activation="relu")))
    model.add((Conv2D(no_filters, size_filter_1, activation="relu")))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add((Conv2D(no_filters // 2, size_filter_2, activation="relu")))
    model.add((Conv2D(no_filters // 2, size_filter_2, activation="relu")))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_nodes, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Create the list of all the pathway to subfolders containing the characters data
path_list = []
print("Creating list of pathways ...")
for root, subdirectories, files in os.walk(path):
    for subdirectory in subdirectories:
        path_list.append(os.path.join(root, subdirectory))
del path_list[:3]

label_list = [label[-1] for label in path_list]
no_of_classes = len(label_list)
print("Number of classes: ", no_of_classes)


# Create the lists of images and matching class_name
images = []
class_name = []
print("Importing and resizing images from dataset ...")
for x in tqdm(range(0, no_of_classes)):
    picture_list = os.listdir(path_list[x])
    print(f"[DEBUG] path is {path_list[x]}")
    for y in picture_list:
        current_image = cv2.imread(path_list[x] + "/" + y)
        if current_image is None:
            print(f"Wrong path for {current_image}")
        else:
            current_image = cv2.resize(current_image, (image_dimension[0], image_dimension[1]))
            images.append(current_image)
            class_name.append(path_list[x][-1])

# Convert images and their labels into np.array
print("Converting images to np.arrays ...")
images = np.array(images)
class_name = np.array(class_name)


# Train, test & validation split + preprocessing
print("Train / test / validation split ...")
X_train, X_test, y_train, y_test = train_test_split(
    images, class_name, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

print("Preprocessing of Train / Test / Validation sets ...")
X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))


# Reshape for CNN
print("Reshaping for CNN ...")
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)


# Image Augmentation to make dataset bigger and more generic
print("Image Data Generator ...")
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                    zoom_range=0.2, shear_range=0.1, rotation_range=10)
data_generator.fit(X_train)


# One Hot Encoding before using to_categorical method cause Keras only takes dtype INT or FLOAT
print("Label One Hot Encoding ...")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(np.array(y_train))
y_test = label_encoder.fit_transform(np.array(y_test))
y_validation = label_encoder.fit_transform(np.array(y_validation))

# Save the classes for later decoding in OCR_CNN_test.py with the same 'hash'
np.save('classes.npy', label_encoder.classes_)

print("Passing to_categorical function from Keras ...")
y_train = to_categorical(y_train, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)
y_validation = to_categorical(y_validation, no_of_classes)


batch_size_value = 2000
epochs_value = 10
steps_per_epoch = len(X_train) // batch_size_value


# Creating the model
model = my_model()
print(model.summary())
history = model.fit(data_generator.flow(X_train, y_train, batch_size=batch_size_value),
                    steps_per_epoch=steps_per_epoch, epochs=epochs_value,
                    validation_data=(X_validation, y_validation), shuffle=1)

# Plot Loss & Accuracy
plt.figure(1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training", "Validation"])
plt.title("Loss")
plt.xlabel("Epochs")
plt.figure(2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["Training", "Validation"])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.show()

# Printing the Score & Accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Our test score is {score[0]}")
print(f"Our test accuracy is {score[1]}")


# Saving the model in a Pickle File ("wb" = Write Bytes)
pickle_out = open("CNN_trained_model", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
