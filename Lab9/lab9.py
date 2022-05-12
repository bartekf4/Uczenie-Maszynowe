#!/usr/bin/env python
# coding: utf-8

# In[134]:


from sklearn.datasets import load_iris

iris = load_iris(as_frame=False)

# In[135]:


# pd.concat([iris.data, iris.target], axis=1).plot.scatter(x='petal length (cm)', y='petal width (cm)', c='target',
#                                                          colormap='viridis')


# Perceptrony i irysy

# In[136]:


from sklearn.model_selection import train_test_split

X = iris.data[:, (2, 3)]
rnd_state = 20
X_train0, X_test0, y_train0, y_test0 = train_test_split(X, (iris.target == 0).astype(int), train_size=.8,
                                                        random_state=rnd_state)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, (iris.target == 1).astype(int), train_size=.8,
                                                        random_state=rnd_state)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, (iris.target == 2).astype(int), train_size=.8,
                                                        random_state=rnd_state)

# In[137]:


from sklearn.linear_model import Perceptron

per_clf0 = Perceptron()
per_clf1 = Perceptron()
per_clf2 = Perceptron()

# In[138]:


per_clf0.fit(X_train0, y_train0)
per_clf1.fit(X_train1, y_train1)
per_clf2.fit(X_train2, y_train2)

X_train2

# In[139]:


per_acc = [(per_clf0.score(X_train0, y_train0), per_clf0.score(X_test0, y_test0)),
           (per_clf1.score(X_train1, y_train1), per_clf1.score(X_test1, y_test0)),
           (per_clf2.score(X_train2, y_train2), per_clf2.score(X_test2, y_test2))]
per_acc

# In[140]:


per_wght = []
for perceptron in [per_clf0, per_clf1, per_clf2]:
    w_0 = perceptron.intercept_[0]
    w_1 = perceptron.coef_[0, 0]
    w_2 = perceptron.coef_[0, 1]
    per_wght.append((w_0, w_1, w_2))
per_wght

# In[141]:


import pickle

with open('per_acc.pkl', 'wb') as fp:
    pickle.dump(per_acc, fp)

with open('per_wght.pkl', 'wb') as fp:
    pickle.dump(per_wght, fp)

# Perceptron i XOR

# In[142]:


import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0,
              1,
              1,
              0])
per_clf_xor = Perceptron()
per_clf_xor.fit(X, y)

# XOR, drugie podejście

# In[191]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

model = Sequential()
model.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
model.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              metrics=["accuracy"])

# In[212]:


found = False
while not found:
    model = Sequential()
    model.add(keras.layers.Dense(2, activation="tanh", use_bias=True, input_dim=2))
    model.add(keras.layers.Dense(1, activation="sigmoid", use_bias=True))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.09),
                  metrics=["accuracy"])
    history = model.fit(X, y, epochs=100, verbose=False)
    results = model.predict(X)
    if 0.1 > results[0] > 0 and 1 > results[1] > 0.9 and 1 > results[2] > 0.9 and 0 < results[3] < 0.1:
        found = True
mlp_xor_weights = model.get_weights()
print(results)
with open('mlp_xor_weights.pkl', 'wb') as file:
    pickle.dump(mlp_xor_weights, file)
