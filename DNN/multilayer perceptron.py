#%%

import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r'c:/Users/Surface/Desktop/tensorflow/')

Data = pd.read_csv('Advertising.csv', header=0, index_col=0)
Data.head()
plt.scatter(Data.TV, Data.sales)
plt.show()

x = Data.iloc[:, 0:-1]
y = Data.iloc[:, -1]
print('x:',type(x),len(x))
print(y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3, ), activation='relu'),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=50)
#%%

test = Data.iloc[:10, 0:-1]
predict=model.predict(test)
Actual = Data.iloc[:10, -1]

print (predict)
print (Actual)















# %%
