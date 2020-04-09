from flask import Flask, request, jsonify
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import json

app = Flask(__name__)

@app.route("/", methods = ['POST'])
def home():



    #get model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # gets payload
    image = np.array(request.json['payload']).astype(np.float32)

    #change colorspace
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # rotate
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, 90, 1)
    image = cv2.warpAffine(image, rot_mat, (h, w))

    # resize to the correct dims
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    imgset = np.array([image]).reshape(1, 28, 28, 1)

    preds = model.predict(imgset)
    print(np.argmax(preds).astype(np.float32))

    return json.dumps(str(np.argmax(preds)))

if __name__ == "__main__":
    app.run()