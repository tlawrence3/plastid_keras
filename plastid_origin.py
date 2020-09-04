import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from sklearn.utils import class_weight
from sklearn import preprocessing
X = []
y = []
plastids = []

with open("taxalist_resample.txt", "r") as taxon_class:
    for line in taxon_class:
        class_name = line.split()[0]
        y.append(class_name)

with open("loocv_resample.txt", "r") as lcv_scores:
    for line in lcv_scores:
        scores = line.split()[1:-1]
        scores = [float(x) for x in scores]
        X.append(scores)

with open("plastid.tRNAs.aligned.final.scores", "r") as plastids_scores:
    for line in plastids_scores:
        scores = line.split()[1:-1]
        scores = [float(x) for x in scores]
        plastids.append(scores)

X = np.array(X)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
#X = (X - np.mean(X, axis = 0, dtype=np.float64)) / np.std(X, axis = 0, dtype=np.float64)
y = np.array(y)
lb_class = preprocessing.LabelEncoder()
lb_class.fit(y)
y = lb_class.transform(y)
plastids = np.array(plastids)
plastids = scaler.transform(plastids)
#plastids = (plastids - np.mean(X, axis = 0, dtype=np.float64)) / np.std(X, axis = 0, dtype=np.float64)

inputs = keras.Input(shape=(8,))
x = layers.Dense(11, activation="relu")(inputs)
x = layers.Dense(14, activation="relu")(x)
outputs = layers.Dense(8, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="plastid_model")
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.fit(X, y, batch_size=240, epochs=1500)

print(lb_class.inverse_transform(model.predict(plastids).argmax(axis=-1)))
