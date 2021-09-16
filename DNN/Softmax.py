#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(Train_image,
 Train_label), (Test_image,
                Test_label) = tf.keras.datasets.fashion_mnist.load_data()

print(Train_image.shape)

Train_image = Train_image / 255
Test_image = Test_image / 255

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
#%%
#One-Hot Encoding

Train_label_OneHot = tf.keras.utils.to_categorical(Train_label)
Test_label_OneHot = tf.keras.utils.to_categorical(Test_label)

Train_label_OneHot

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(Train_image,
                    Train_label_OneHot,
                    epochs=20,
                    validation_data=(Test_image, Test_label_OneHot))

plt.plot(history.epoch,
         history.history.get('loss'),
         label='loss',
         color='red')
plt.plot(history.epoch,
         history.history.get('val_loss'),
         label='val_loss',
         color='blue')
model.evaluate(Test_image, Test_label_OneHot)
#%%

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(Train_image, Train_label, epochs=5)

model.evaluate(Test_image, Test_label)

# %%
