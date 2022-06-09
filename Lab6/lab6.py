#!/usr/bin/env python
# coding: utf-8

# Podziel zbiór data_breast_cancer na uczący i testujący w proporcjach 80:20

# In[63]:


import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data_breast_cancer = load_breast_cancer(as_frame=True)

# Dzielimy na zbiór uczący i testowy (pod uwagę bierzemy cechy 'mean texture', 'mean symmetry' )

# In[64]:


X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer['data'][['mean texture', 'mean symmetry']],
                                                    data_breast_cancer['target'],
                                                    test_size=.2)

# In[65]:


from sklearn.tree import DecisionTreeClassifier

dec_tree_clf = DecisionTreeClassifier()
dec_tree_clf.fit(X_train, y_train)

# In[66]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# In[67]:


from sklearn.neighbors import KNeighborsClassifier

knn_reg = KNeighborsClassifier()
knn_reg.fit(X_train, y_train)

# In[68]:


from sklearn.ensemble import VotingClassifier

voting_hard_clf = VotingClassifier(
    estimators=[
        ('dec_tree', dec_tree_clf),
        ('log_reg', log_reg),
        ('knn_clf', knn_reg)],
    voting='hard'
)

voting_hard_clf.fit(X_train, y_train)

voting_soft_clf = VotingClassifier(
    estimators=[
        ('dec_tree', dec_tree_clf),
        ('log_reg', log_reg),
        ('knn_clf', knn_reg)],
    voting='soft'
)
voting_soft_clf.fit(X_train, y_train)

# In[69]:


acc_vote = [(dec_tree_clf.score(X_train, y_train), dec_tree_clf.score(X_test, y_test)),
            (log_reg.score(X_train, y_train), log_reg.score(X_test, y_test)),
            (knn_reg.score(X_train, y_train), knn_reg.score(X_test, y_test)),
            (voting_hard_clf.score(X_train, y_train), voting_hard_clf.score(X_test, y_test)),
            (voting_soft_clf.score(X_train, y_train), voting_soft_clf.score(X_test, y_test))]

# piklowanie acc_vote

# In[70]:


import pickle

with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(acc_vote, f)

# In[71]:


with open('acc_vote.pkl', 'rb') as f:
    print(pickle.load(f))

# piklowanie klasyfikatorów

# In[72]:


vote = [dec_tree_clf,
        log_reg,
        knn_reg,
        voting_hard_clf,
        voting_soft_clf]

with open('vote.pkl', 'wb') as f:
    pickle.dump(vote, f)

# In[73]:


with open('vote.pkl', 'rb') as f:
    print(pickle.load(f))

# Wykonaj na zbiorze uczącym wykorzystując 30 drzew decyzyjnych

# In[74]:


from sklearn.ensemble import BaggingClassifier

bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                n_estimators=30)
bagging_clf.fit(X_train, y_train)

# In[75]:


bagging_50_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                   n_estimators=30,
                                   max_samples=0.5)
bagging_50_clf.fit(X_train, y_train)

# In[76]:


pasting_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                n_estimators=30,
                                bootstrap=False)
pasting_clf.fit(X_train, y_train)

# In[77]:


pasting_50_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                   n_estimators=30,
                                   max_samples=0.5,
                                   bootstrap=False)
pasting_50_clf.fit(X_train, y_train)

# In[78]:


from sklearn.ensemble import RandomForestClassifier

ran_for_clf = RandomForestClassifier(n_estimators=30)
ran_for_clf.fit(X_train, y_train)

# In[79]:


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators=30)
ada_clf.fit(X_train, y_train)

# In[80]:


from sklearn.ensemble import GradientBoostingClassifier

grad_clf = GradientBoostingClassifier(n_estimators=30)
grad_clf.fit(X_train, y_train)

# In[81]:


acc_bag = [(bagging_clf.score(X_train, y_train), bagging_clf.score(X_test, y_test)),
           (bagging_50_clf.score(X_train, y_train), bagging_50_clf.score(X_test, y_test)),
           (pasting_clf.score(X_train, y_train), pasting_clf.score(X_test, y_test)),
           (pasting_50_clf.score(X_train, y_train), pasting_50_clf.score(X_test, y_test)),
           (ran_for_clf.score(X_train, y_train), ran_for_clf.score(X_test, y_test)),
           (ada_clf.score(X_train, y_train), ada_clf.score(X_test, y_test)),
           (grad_clf.score(X_train, y_train), grad_clf.score(X_test, y_test))]

acc_bag

# In[82]:


with open("acc_bag.pkl", 'wb') as f:
    pickle.dump(acc_bag, f)
with open("acc_bag.pkl", 'rb') as f:
    print(pickle.load(f))

# In[83]:


bag = [bagging_clf, bagging_50_clf, pasting_clf, pasting_50_clf, ran_for_clf, ada_clf, grad_clf]
bag

# In[84]:


with open('bag.pkl', 'wb') as f:
    pickle.dump(bag, f)

with open('bag.pkl', 'rb') as f:
    print(pickle.load(f))

# Przeprowadź sampling 2 cech z wszystkich dostepnych bez powtórzeń z wykorzystaniem 30 drzew decyzyjnych, wybierz połowę instancji dla każdego z drzew z powtórzeniami.

# In[85]:


X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer['data'],
                                                    data_breast_cancer['target'], test_size=.2, random_state=5)

# In[86]:


bag_2_features = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=30, max_features=2,
                                   bootstrap_features=False,
                                   bootstrap=True, max_samples=.5, random_state=25)
bag_2_features.fit(X_train, y_train)

# Zapisz dokładności ww estymatora listę : dokładność_dla_zb_uczącego, dokładność_dla_zb_testującego w pliku Pickle acc_fea.pkl.
# 

# In[87]:


acc_fea = [bag_2_features.score(X_train, y_train), bag_2_features.score(X_test, y_test)]
acc_fea

# In[88]:


with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(acc_fea, f)

with open('acc_fea.pkl', 'rb') as f:
    print(pickle.load(f))

# Zapisz klasyfikator jako jednoelementową listę w pliku Pickle o nazwie fea.pkl

# In[89]:


with open('fea.pkl', 'wb') as f:
    pickle.dump(bag_2_features, f)

with open('fea.pkl', 'rb') as f:
    print(pickle.load(f))

#

# Sprawdź, które cechy dają najwięszą dokładność. Dostęp do poszczególnych estymatorów,
# aby obliczyć dokładność, możesz uzyskać za pmocą: BaggingClasifier.estimators_,
# cechy wybrane przez sampling dla każdego z estymatorów znajdziesz w:
# BaggingClassifier.estimators_features_. Zbuduj ranking estymatorów jako DataFrame,
# który będzie mieć w kolejnych kolumnach: dokładność dla zb. uczącego, dokładnośc dla zb.
# testującego, lista nazw cech. Każdy wiersz to informacje o jednym estymatorze. DataFrame
# posortuj malejąco po wartościach dokładności dla zbioru testującego i uczącego oraz zapisz
# w pliku Pickle o nazwie acc_fea_rank.pkl

# In[90]:


features_array = []
for features in bag_2_features.estimators_features_:
    features_array.append([data_breast_cancer["feature_names"][features[0]],
                           data_breast_cancer["feature_names"][features[1]]])

# In[91]:


acc_fea_rank = []
for estimator, features in zip(bag_2_features.estimators_, features_array):
    acc_fea_rank.append(
        [estimator.score(X_train[features], y_train), estimator.score(X_test[features], y_test), features])

# In[92]:


df_rank = pd.DataFrame(acc_fea_rank)
df_rank.sort_values(inplace=True, by=[0, 1], ascending=False)
df_rank

# In[93]:


df_rank.to_pickle("acc_fea_rank.pkl")

# In[93]:
