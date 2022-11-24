

from os import listdir
import cv2
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

import matplotlib.pyplot as plt

import random

data_folder = "/Users/haidang/Documents/CNN_Fruit/Fruit_Data/Training/"

def save_data(data_folder=data_folder):

    img_size = (224, 224)
    print("Bắt đầu thêm ảnh...")

    images = []
    labels = []
    names = []

    # Đọc các thư mục con trong folder data
    for folder in listdir(data_folder):
          print(folder)
          names.append(folder)
            # Đọc các file trong thư mục con và lưu hình vào images, nhãn vào label
          for file in listdir(data_folder  + folder):
            try:
                images.append(cv2.resize(cv2.imread(data_folder  + folder +"/" + file),dsize=(224,224)))
                labels.append(folder)
            except Exception as e:
                print(str(e))
                print(file)

    images = np.array(images)
    labels = np.array(labels)#.reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)

    file1 = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-train.data','wb')
    file2 = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-train-class.data','wb')
    # dump information to that file
    pickle.dump((images,labels), file1)
    pickle.dump(names, file2)
    # close the file
    file1.close()
    file2.close()

    return

save_data()

def load_data():
    file = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-train.data', 'rb')

    # dump information to that file
    (images, labels) = pickle.load(file)

    # close the file
    file.close()

    print(images.shape)
    print(labels.shape)


    return images, labels

test_folder = "/Users/haidang/Documents/CNN_Fruit/Fruit_Data/Test/"

def save_test_data(test_folder=test_folder):

    img_size = (224, 224)
    print("Bắt đầu thêm ảnh...")

    images = []
    labels = []
    names = []

    # Đọc các thư mục con trong folder data
    for folder in listdir(test_folder):
            print(folder)
            names.append(folder)
            # Đọc các file trong thư mục con và lưu hình vào images, nhãn vào label
            for file in listdir(test_folder  + folder):
              try:
                    images.append(cv2.resize(cv2.imread(test_folder  + folder +"/" + file),dsize=(224,224)))
                    labels.append(folder)
              except Exception as e:
                print(str(e))
                print(file)

    images = np.array(images)
    labels = np.array(labels)#.reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)

    file1 = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-test.data','wb')
    file2 = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-test-class.data','wb')
    # dump information to that file
    pickle.dump((images,labels), file1)
    pickle.dump(names, file2)
    # close the file
    file1.close()
    file2.close()

    return

save_test_data()

def load_test_data():
    file = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-test.data', 'rb')

    # dump information to that file
    (images, labels) = pickle.load(file)

    # close the file
    file.close()

    print(images.shape)
    print(labels.shape)

    return images, labels

def load_class():
    file = open('/Users/haidang/Documents/CNN_Fruit/Model/fruit-test-class.data', 'rb')

    # dump information to that file
    names = pickle.load(file)

    # close the file
    file.close()

    print(len(names))

    return names

X,y = load_data()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=100)

print(X_train.shape)
print(y_train.shape)

def convertImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

X_train = np.array(list(map(convertImage, X_train)))
X_test = np.array(list(map(convertImage, X_test)))

X_train = X_train.reshape(-1, 224, 224, 3)
X_test = X_test.reshape(-1, 224, 224, 3)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

import keras

#there are using maxpool convolution and final dense layer.
model = Sequential()
# First convolutional layer, note the specification of shape
model.add(Conv2D(128, 3, activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(Dense(30, activation = "softmax"))
model.summary()
model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=[keras.metrics.BinaryAccuracy()])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)
model.save("/Users/haidang/Documents/CNN_Fruit/Model/fruit-classifier.h5")


