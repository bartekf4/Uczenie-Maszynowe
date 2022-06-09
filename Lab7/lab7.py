#!/usr/bin/env python
# coding: utf-8

# # Przygotowanie danych

# In[1]:


import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

# In[2]:


from sklearn.cluster import KMeans

list_of_kmeans = []
list_of_y_pred = []

for i in range(8, 13, 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    y_pred = kmeans.fit_predict(X, y)

    list_of_y_pred.append(y_pred)
    list_of_kmeans.append(kmeans)

# In[3]:


from sklearn.metrics import silhouette_score

kmeans_sil = []
for kmean in list_of_kmeans:
    kmeans_sil.append(silhouette_score(X, kmean.labels_))

# In[4]:


import pickle

with open('kmeans_sil.pkl', 'wb') as file:
    pickle.dump(kmeans_sil, file)

with open('kmeans_sil.pkl', 'rb') as file:
    print(pickle.load(file))

# In[23]:


from sklearn.metrics import confusion_matrix

con_mat = confusion_matrix(y, list_of_y_pred[2])

# In[24]:


kmeans_argmax = set()

for row in con_mat:
    print(row)
    kmeans_argmax.add(np.argmax(row))

# In[25]:


with open('kmeans_argmax.pkl', 'wb') as file:
    pickle.dump(kmeans_argmax, file)

with open('kmeans_argmax.pkl', 'rb') as file:
    print(pickle.load(file))

# ## DBSCAN

# In[ ]:


no_of_samples = 300

dist = [np.linalg.norm(X[i] - x2) for i in range(300) for x2 in X if not (X[i] == x2).all()]

dist.sort()
dist = dist[:10]

# In[ ]:


with open('dist.pkl', 'wb') as file:
    pickle.dump(dist, file)

with open('dist.pkl', 'rb') as file:
    print(pickle.load(file))

# In[ ]:


mean = np.mean(dist[:3])

# In[ ]:


from sklearn.cluster import DBSCAN

dbscan_len = []
for eps in np.arange(mean, mean * 1.1, mean * 0.04):
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)
    dbscan_len.append(len(set(dbscan.labels_)))

# In[ ]:


with open('dbscan_len.pkl', 'wb') as file:
    pickle.dump(dbscan_len, file)

# In[ ]:
