#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r'c:/Users/Surface/Desktop/tensorflow/')

Data = pd.read_csv('Income1.csv', header=0, index_col=0)
print(Data)
plt.scatter(Data.Education, Data.Income)
plt.show()

x = Data.Education
y = Data.Income

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1, )))
print(model.summary())
model.compile(optimizer='adam', loss='mse') 
model.fit(x,y,epochs=1000)

model.predict(pd.Series([20]))

# %%
