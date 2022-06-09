#!/usr/bin/env python
# coding: utf-8

# Pobierz zbiór danych Fashion MNIST.

# In[140]:


import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Wyświetl przykładowy rysunek używany do klasyfikacji:

# In[141]:


import matplotlib.pyplot as plt

plt.imshow(X_train[142], cmap="binary")
plt.axis('off')
plt.show()

# In[142]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[142]]

# In[143]:


len(X_train)

# In[143]:


# In[144]:


import numpy as np

X_train = np.array(X_train) / 255
X_test = np.array(X_test) / 255

# In[145]:


from tensorflow import keras

model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(10, activation="softmax"))

# In[146]:


model.summary()
keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)

# Skompiluj model, podając rzadką entropię krzyżową jako funkcję straty, SGD jako opymalizator i
# dokładność jako metrykę

# In[147]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# In[148]:


import os

root_logdir = os.path.join(os.curdir, "image_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# In[149]:


checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "my_keras_model.h5", save_best_only=True)
model.fit(X_train, y_train, epochs=20, validation_split=.1, callbacks=[checkpoint_cb, tensorboard_cb])

# In[150]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()

# Zapisz model w pliku fashion_clf.h5:
# 

# In[151]:


model.save('fashion_clf.h5')

# Pobierz zbiór danych California Housing z pakietu scikit-learn:
# 

# In[200]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

# In[199]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42, shuffle=True)

# Przeskaluj wszystkie zbiory cech, kalibrując funkcję normalizacyjną do zbioru uczącego:
# 

# In[224]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# In[225]:


reg_housing_1 = keras.models.Sequential()
reg_housing_1.add(keras.layers.Dense(30, input_shape=[X_train.shape[1]], activation='relu'))
reg_housing_1.add(keras.layers.Dense(1))

# Skompiluj go używając błędu średniokwadratwego jako funkcji straty i SGD jako optymalizatora.

# In[226]:


reg_housing_1.compile(loss='mean_squared_error', optimizer='sgd')

# Przygotuj callback early stopping o cierpliwości równej 5 epok, minimalnej wartości poprawy
# wynoszącej 0.01 i włączając wyświetlanie komunikatów o przerwaniu uczenia na ekranie

# In[227]:


es = tf.keras.callbacks.EarlyStopping(patience=5,
                                      min_delta=0.01,
                                      verbose=1)

#

# Podobnie jak w poprzednim ćwiczeniu, przygotuj callback Tensorboard, tak aby zbierał logi do
# katalogu housing_logs.

# In[228]:


root_logdir = os.path.join(os.curdir, "housing_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard = tf.keras.callbacks.TensorBoard(run_logdir)

# In[216]:


reg_housing_1.summary()

# Skompiluj go używając błędu średniokwadratwego jako funkcji straty i SGD jako optymalizatora.

# In[229]:


history = reg_housing_1.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid),
                            callbacks=[es, tensorboard])

# In[230]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

# In[231]:


model.save("reg_housing_1.h5")

# In[232]:


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', kernel_initializer='normal', kernel_regularizer="l2",
                       input_shape=[X_train.shape[1]]),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(64, activation='relu', kernel_initializer='normal', kernel_regularizer="l2"),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(1)
])

model.compile(loss='mae',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=64,
    validation_data=(X_valid, y_valid),

)

# In[233]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

# In[235]:


model.save("reg_housing_2.h5")

# In[243]:


model = keras.Sequential([
    keras.layers.Dense(10, activation='softmax',
                       input_shape=[X_train.shape[1]]),

    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),

    keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer="SGD",
              metrics=['mse'])

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_valid, y_valid),

)

# In[244]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

# In[ ]:


model.save("reg_housing_3.h5")
