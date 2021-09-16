#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r'c:/Users/Surface/Desktop/tensorflow/')

Data = pd.read_csv('credit-a.csv', header=None)

Data.head()

x = Data.iloc[:, :-1]
y = Data.iloc[:, -1].replace(-1, 0)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(4, input_shape=(15, ), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x, y, epochs=1000)

# %%
history.history.keys()

plt.plot(history.epoch, history.history.get('loss'))




# %%
plt.plot(history.epoch, history.history.get('acc'))
# %%
