import os
import pandas as pd
import numpy as np
import ta
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputdata = pd.read_csv("Weatherdata.csv")
outputdata = pd.read_csv("Energydata.csv")

inputdata["dt_iso"] = pd.to_datetime(inputdata.dt_iso)
inputdata.set_index('dt_iso', inplace=True)
inputdata.dropna(inplace=True)

outputdata.dropna(inplace=True)

inputdata = np.array(inputdata)
labels = np.array(outputdata)

print(np.shape(inputdata))
print(np.shape(labels))

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(inputdata.shape[1])))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(400, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(60, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(13, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy'],
)

print(model.summary())

print(np.shape(inputdata))
print(np.shape(outputdata))

history = model.fit(inputdata, labels, epochs=50, batch_size=32, validation_split=0.2)

model.save('model.h5')
