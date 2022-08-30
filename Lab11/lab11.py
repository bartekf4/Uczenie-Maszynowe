#!/usr/bin/env python
# !/usr/bin/env python
# coding: utf-8

# # Strojenie hiperparametrów

# ## Zadania

# ### Poszukiwanie ręczne

# Pobierz zestaw danych Boston Housing:

# In[105]:


import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# Przygotuj funkcję budującą model według parametrów podanych jako argumenty:
# - n_hidden – liczba warstw ukrytych,
# - n_neurons – liczba neuronów na każdej z warstw ukrytych,
# - optimizer – gradientowy algorytm optymalizacji, funkcja powinna rozumieć wartości: sgd, nesterov, momentum oraz adam,
# - learning_rate – krok uczenia,
# - momentum – współczynnik przyspieszenia dla algorytmów z pędem.

# In[106]:


X_train


# In[107]:


def build_model(optimizer, n_hidden=1, learning_rate=10e-5, n_neurons=25, momentum=0):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    if optimizer == "momentum":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae'])
    return model


# Przy uczeniu wykorzystaj mechanizm early stopping o cierpliwości równej 10 i minimalnej poprawie
# funkcji straty równej 1.00, uczenie maksymalnie przez 100 epok.

# In[108]:


es = tf.keras.callbacks.EarlyStopping(patience=10,
                                      min_delta=1.00)


# In[109]:


def get_run_logdir(name, value):
    import time
    import os
    root_logdir = os.path.join(os.curdir, "tb_logs")
    ts = int(time.time())

    run_id = str(ts) + "_" + name + "_" + str(value)
    return os.path.join(root_logdir, run_id)


# Przed eksperymentami wyczyść sesję TensorFlow i ustal generatory liczb losowych:

# In[110]:


import numpy as np

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# In[111]:


epochs = 100
validation_split = .1

# krok uczenia(lr): 10−6, 10−5, 10−4

# In[112]:


results_lr = []
for lr in (10e-6, 10e-5, 10e-4):
    run_logdir = get_run_logdir("lr", lr)
    tensorboard = tf.keras.callbacks.TensorBoard(run_logdir)
    model = build_model(learning_rate=lr, optimizer="sgd")
    model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[es, tensorboard])
    score = model.evaluate(X_test, y_test)
    results_lr.append((lr, score[0], score[1]))

# In[113]:


results_lr

# liczba warstw ukrytych (hl): od 0 do 3,

# 

# In[114]:


results_hl = []
for hl in (0, 1, 2, 3):
    run_logdir = get_run_logdir("hl", hl)
    tensorboard = tf.keras.callbacks.TensorBoard(run_logdir)
    model = build_model(n_hidden=hl, optimizer="sgd")
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[es, tensorboard])
    score = model.evaluate(X_test, y_test)
    results_hl.append((hl, score[0], score[1]))

# In[115]:


results_hl

# liczba neuronów na warstwę (nn): 5, 25, 125

# In[116]:


results_nn = []
for nn in (5, 25, 125):
    run_logdir = get_run_logdir("nn", nn)
    tensorboard = tf.keras.callbacks.TensorBoard(run_logdir)
    model = build_model(optimizer="sgd", n_neurons=nn)
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[es, tensorboard])
    score = model.evaluate(X_test, y_test)
    results_nn.append((nn, score[0], score[1]))

# In[117]:


results_nn

# algorytm optymalizacji (opt): wszystkie 4 algorytmy (pęd = 0.5)

# In[118]:


results_alg = []
for alg in ("sgd", "nesterov", "momentum", "adam"):
    run_logdir = get_run_logdir("alg", alg)
    tensorboard = tf.keras.callbacks.TensorBoard(run_logdir)
    model = build_model(optimizer=alg, momentum=.5)
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[es, tensorboard])
    score = model.evaluate(X_test, y_test)
    results_alg.append((alg, score[0], score[1]))

# In[119]:


results_alg

#  pęd (mom): 0.1, 0.5, 0.9 (dla algorytmu momentum).

# In[120]:


results_mom = []
for mom in (.1, .5, .9):
    run_logdir = get_run_logdir("momentum", mom)
    tensorboard = tf.keras.callbacks.TensorBoard(run_logdir)
    model = build_model(optimizer="momentum", momentum=mom)
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[es, tensorboard])
    score = model.evaluate(X_test, y_test)
    results_mom.append((mom, score[0], score[1]))

# In[121]:


results_mom

# ### Automatyczne poszukiwanie przestrzeni argumentów

# Przygotuj słownik zawierający przeszukiwane wartości parametrów

# In[138]:


param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": [5, 25, 125],
    "learning_rate": [10e-6, 10e-5, 10e-4],
    "optimizer": ["sgd", "adam", "nesterov", "momentum"],
    "momentum": [.1, .5, .9]
}

# Przygotuj callback early stopping i obuduj przygotowaną wcześniej funkcję build_model obiektem
# KerasRegressor

# In[140]:


from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])

# In[147]:


from sklearn.model_selection import RandomizedSearchCV

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=30, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Zapisz najlepsze znalezione parametry w postaci słownika do pliku rnd_search.pkl.

# In[148]:


rnd_search_cv.best_params_

# In[146]:


# In[149]:


import pickle

with open('lr.pkl', 'wb') as file:
    pickle.dump(results_lr, file)

with open('hl.pkl', 'wb') as file:
    pickle.dump(results_hl, file)

with open('nn.pkl', 'wb') as file:
    pickle.dump(results_nn, file)

with open('opt.pkl', 'wb') as file:
    pickle.dump(results_alg, file)

with open('mom.pkl', 'wb') as file:
    pickle.dump(results_mom, file)

with open('rnd_search.pkl', 'wb') as file:
    pickle.dump(rnd_search_cv.best_params_, file)

# In[155]:
