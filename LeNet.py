import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import matplotlib.backends.backend_qt5agg
from matplotlib.figure import Figure
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.datasets import mnist
import matplotlib.image as img
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_test = x_test
X_train = x_train
Y_train = y_train
Y_test = y_test
#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = np.pad(array=x_train, pad_width=((0,0),(1,1),(1,1)), mode='constant', constant_values=0)
x_test = np.pad(array=x_test, pad_width=((0,0),(1,1),(1,1)), mode='constant', constant_values=0)
x_train = np.pad(array=x_train, pad_width=((0,0),(1,1),(1,1)), mode='constant', constant_values=0)
x_test = np.pad(array=x_test, pad_width=((0,0),(1,1),(1,1)), mode='constant', constant_values=0)
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#score = model.evaluate(x_test,y_test)

#print("\ntest score",score[0])
#print("test accuracy",score[1])

class Ui_test(object):
    def button_5_1(self):
        num = []
        i = 0
        j = 0
        random.seed()
        while i < 10:
            index = random.randint(0,(len(X_train)-1))
            if(index not in num):
                num.append(index)
                i=i+1
        plt.figure()
        for i in num:
            plt.subplot(1,10,j+1)
            j = j + 1
            image = X_train[i].squeeze()
            plt.xlabel(Y_train[i])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image, cmap='gray')
        plt.show()
    def button_5_2(self):
        print("hyperparameters:")
        print("batch size: 32")
        print("learning rate: 0.001")
        print("optimizer: SGD")

    def button_5_3(self):
        sgd = tf.keras.optimizers.SGD(lr = 0.001)
        model = tf.keras.models.Sequential()

        #conv -> relu -> pool
        model.add(tf.keras.layers.Conv2D(6,kernel_size=5,strides=(1, 1),padding="valid",input_shape=(32,32,1)))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        #conv -> relu -> pool
        model.add(tf.keras.layers.Conv2D(16,kernel_size=5,strides=(1, 1),padding="valid",input_shape=(14,14,1)))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        #conv -> relu -> pool
        model.add(tf.keras.layers.Conv2D(120,kernel_size=5,strides=(1, 1),padding="valid",input_shape=(5,5,1)))
        model.add(tf.keras.layers.Activation("relu"))


        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(84))
        model.add(tf.keras.layers.Activation("relu"))

        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Activation("softmax"))

        model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=['accuracy'])
        losshistory = LossHistory()
        history = model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(x_test, y_test),
                            callbacks=[losshistory])
        x_value = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        x_value = x_value.flatten()
        plt.xticks(x_value)
        plt.title('1 Epoch')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.plot(losshistory.losses)
        plt.show()

    def button_5_4(self):
        model = tf.keras.models.load_model('Lenet_Model')
        data = self.inputtext.text()
        input = x_test[int(data)]
        input= input[np.newaxis,]
        prediction = model.predict(input)
        value = np.array([0,1,2,3,4,5,6,7,8,9])
        value = value.flatten()
        y_value = np.array([0,0.2,0.4,0.6,0.8,1.0])
        y_value = y_value.flatten();
        prediction = prediction.flatten()
        plt.figure();
        plt.subplot(2,1,1)
        plt.imshow(X_test[int(data)], cmap='gray')
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.xticks(value)
        plt.yticks(y_value)
        plt.bar(value,prediction)
        plt.show()
    def setupUi(self, test):
        test.setObjectName("test")
        test.resize(606, 449)

        self.centralwidget = QtWidgets.QWidget(test)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(180, 60, 201, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(180, 120, 201, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(180, 180, 201, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(180, 240, 201, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.inputtext = QtWidgets.QLineEdit(self.centralwidget)
        self.inputtext.setGeometry(QtCore.QRect(260, 300, 121, 31))
        self.inputtext.setObjectName("inputtext")
        #self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        #self.textEdit.setGeometry(QtCore.QRect(260, 300, 121, 31))
        #self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 300, 121, 31))
        self.label.setObjectName("label")
        test.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(test)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 606, 25))
        self.menubar.setObjectName("menubar")
        test.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(test)
        self.statusbar.setObjectName("statusbar")
        test.setStatusBar(self.statusbar)
        self.inputtext.setPlaceholderText('(0~9999)')
        self.inputtext.setAlignment(QtCore.Qt.AlignRight )
        self.retranslateUi(test)
        self.pushButton.clicked.connect(self.button_5_1)
        self.pushButton_2.clicked.connect(self.button_5_2)
        self.pushButton_3.clicked.connect(self.button_5_3)
        self.pushButton_4.clicked.connect(self.button_5_4)
        QtCore.QMetaObject.connectSlotsByName(test)

    def retranslateUi(self, test):
        _translate = QtCore.QCoreApplication.translate
        test.setWindowTitle(_translate("test", "MainWindow"))
        self.pushButton.setText(_translate("test", "5.1 Show Train Images"))
        self.pushButton_2.setText(_translate("test", "5.2 Show Hyperparameters"))
        self.pushButton_3.setText(_translate("test", "5.3 Train 1 Epoch"))
        self.pushButton_4.setText(_translate("test", "5.4 Inference"))
        self.label.setText(_translate("test", "Testing Image Index:"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_test();
    mainWindows = QMainWindow()
    # w.resize(400,300)

    # w.move(500,250)
    ui.setupUi(mainWindows)

    mainWindows.show()

    sys.exit(app.exec())


