from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import os
import numpy as np
from malapi import deploy
import cv2
import json


class Model:

    model_name = "model.json"

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def print_shape(self):
        print("shape of Data: ", self.x_train.shape)
        print("shape of image: ", self.x_train[0].shape)

    def show_image(self, idx=0):
        print(self.x_train[idx].shape)
        plt.imshow(self.x_train[idx], cmap='gray')
        plt.show()

    def normalise_data(self):
        # reshape train data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

        # reshape labels into oneHot encoding
        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test = np_utils.to_categorical(self.y_test, 10)

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        if os.path.exists(self.model_name):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        else:
            self.train_model()

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=3, shuffle=True,
                       batch_size=128)

        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    def run_model(self, predict):
        if os.path.exists(self.model_name):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        else:
            self.train_model()

        print(self.x_test[:predict].shape)
        preds = self.model.predict(self.x_test[:predict])

        for idx in range(0,predict):
            pred_result = np.argmax(preds[idx])
            real_result = np.argmax(self.y_test[idx])

            if pred_result == real_result:
                print("Real: {}, Pred: {}  -> CORRECT".format(real_result, pred_result))
            else:
                print("Real: {}, Pred: {}  -> FALSE".format(real_result, pred_result))

    @deploy
    def predict(self, request):
        print("load Model")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        print("get request")
        # request from Unity
        image = np.array(request["payload"]).astype(np.float32)

        # rbg to gray
        print("changing color space")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #rotate
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, 90, 1)
        image = cv2.warpAffine(image, rot_mat, (h, w))

        # resize to the correct dims
        image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)
        imgset = np.array([image]).reshape(1, 28, 28, 1)
        print(imgset.shape)

        # predict
        preds = self.model.predict(imgset)
        print(np.argmax(preds).astype(np.float32))

        return json.dumps(str(np.argmax(preds)))

if __name__ == "__main__":
    model = Model()
    model.print_shape()
    #model.show_image(23)
    model.normalise_data()
    model.build_model()

    model.run_model(30)

