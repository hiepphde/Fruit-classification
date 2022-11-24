# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtGui import *
from PIL import Image, ImageOps
import tkinter.filedialog as fd
import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import sys

root = tk.Tk()
root.withdraw()




class Ui_Form(object):
    model = tf.keras.models.load_model('E:/Study/NoronNhanTao/Fruit/fruit-classifier.h5', compile=False)
    catalog = ['durian', 'apples', ' apricots', ' avocados', ' bananas', ' blackberries', ' blueberries', ' cantaloupes', ' cherries',
               'coconuts', ' figs', ' grapefruits', ' grapes', ' guava', ' kiwifruit', ' lemons', ' limes', ' mangos', ' olives', ' oranges',
               'passionfruit', ' peaches', ' pears', ' pineapples', ' plums', ' pomegranates', ' raspberries', ' strawberries', ' corn', ' watermelons']

    def getImage(self):
        nameimg = fd.askopenfilename(initialdir="/Users/haidang/", title="Select Image File",filetypes=(("All Files", "*.*"), ("JPG File", "*.jpg"), ("PNG File", "*.png")))
        image = Image.open(nameimg)
        self.label_4.setGeometry(QtCore.QRect(40, 40, 224, 224))
        self.label_4.setStyleSheet("border-image:url(" + nameimg + ")")
        self.predict(image)

    def showDia(self):
        dialog =QMessageBox(Form)
        dialog.setWindowTitle("Ban co muon su dung video ?")


    def predict(self,image):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = self.model.predict(data)
        # accuracy = round((max(prediction[0])),2)
        accuracy = float("{:.2f}".format(max(prediction[0])))
        transform = [1 if x >= 0.5 else 0 for x in prediction[0]]
        pred = transform.index(max(transform))
        title = self.catalog[pred]
        pred_lbl = title + " - " + str(accuracy*100) + "%"
        self.label_3.setText(pred_lbl)

    def displayImage(self, img,window=1):
        img=cv2.resize(img,(340,224))
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if (img.shape[2])==4:
                qformat =QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        img = QImage(img,img.shape[1],img.shape[0],qformat)
        img =img.rgbSwapped()
        self.label_4.setPixmap(QPixmap.fromImage(img))
        self.label_4.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def video(self):
        self.label_4.setGeometry(QtCore.QRect(15,40, 340, 224))
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        while True:
            ret, frame = cam.read()
            img_name = "Image_Cam/opencv_frame_{}.png".format(0)
            cv2.imwrite(img_name, frame)
            image = Image.open(img_name)
            self.predict(image)
            self.displayImage(frame,1)
            k= cv2.waitKey(1)
            if k % 256 == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
    def camera(self):
        self.label_4.setGeometry(QtCore.QRect(15, 40, 340, 224))
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        while True:
            ret, frame = cam.read()
            img_name = "Image_Cam/opencv_frame_{}.png".format(0)
            cv2.imwrite(img_name, frame)
            self.displayImage(frame, 1)
            k = cv2.waitKey(1)
            if k % 256 == 32:
                break
        image = Image.open(img_name)
        self.predict(image)
        cam.release()
        cv2.destroyAllWindows()




    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(600, 400)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 600, 400))
        self.label.setStyleSheet("border-image:url(GUI/Image/Group 6.png)")
        self.label.setText("")
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(360, 210, 181, 41))
        self.pushButton.setStyleSheet("QPushButton#pushButton {border-image:url(GUI/Image/Group 1.png);}QPushButton#pushButton:hover {border-image:url(GUI/Image/Group 1 (1).png);}")
        self.pushButton.setText("")
        self.pushButton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.clicked.connect(self.getImage)
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(360, 270, 181, 41))
        self.pushButton_2.setStyleSheet("QPushButton#pushButton_2 {border-image:url(GUI/Image/Group 2.png);}QPushButton#pushButton_2:hover {border-image:url(GUI/Image/Group 2 (1).png);}")
        self.pushButton_2.setText("")
        self.pushButton_2.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.clicked.connect(self.camera)
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(360, 330, 181, 41))
        self.pushButton_3.setStyleSheet("QPushButton#pushButton_3 {border-image:url(GUI/Image/Group 2 (3).png);}QPushButton#pushButton_3:hover {border-image:url(GUI/Image/Group 2 (2).png);}")
        self.pushButton_3.setText("")
        self.pushButton_3.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_3.clicked.connect(self.video)
        self.pushButton_3.setObjectName("pushButton_3")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(10, 290, 311, 41))
        self.label_2.setStyleSheet("border-image:url(GUI/Image/Group 4.png)")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(130, 285, 191, 40))
        self.label_3.setStyleSheet("color:rgb(0, 0, 0); font: 87 18pt \"Segoe UI Black\";")
        self.label_3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(40, 40, 224, 224))
        self.label_4.setStyleSheet("border-image:url(GUI/Image/doodle_fruit 1.png)")
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))


# import res_rc


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())