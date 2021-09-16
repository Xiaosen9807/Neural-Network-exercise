#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

(Train_image,
 Train_label), (Test_image,
                Test_label) = tf.keras.datasets.fashion_mnist.load_data()

New_Input1 = keras.Input(shape=(28, 28))
New_Input2 = keras.Input(shape=(28, 28))

Input_layer1 = keras.layers.Flatten()(New_Input1)
Input_layer2 = keras.layers.Flatten()(New_Input2)

Hidden_layer = keras.layers.concatenate([Input_layer1, Input_layer2])

Hidden_layer = keras.layers.Dense(32, activation='relu')(Hidden_layer)

Hidden_layer = keras.layers.Dropout(0.5)(Hidden_layer)

Hidden_layer = keras.layers.Dense(64, activation='relu')(Hidden_layer)

New_Output = keras.layers.Dense(10, activation='relu')(Hidden_layer)

model = keras.Model(inputs=[New_Input1, New_Input2], outputs=New_Output)

model.summary()
# %%
