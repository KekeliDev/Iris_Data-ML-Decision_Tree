# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 07:48:03 2020

@author: AAARON KOFI GAYI
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# loading data from sklearn
iris_data = pd.read_csv('datasets_19_420_Iris.csv')

#getting labal and features
X = iris_data.data
y = iris_data.target

#splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =  0.7, random_state =  0)

# standardizing data
sc =  StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

#choosing algorithm; decision tree
tree =DecisionTreeClassifier(criterion="entropy", max_depth=7, random_state=3)

#training model
tree.fit(X_train_std,y_train)
y_pred = tree.predict(X_test_std)

print("wrong prediction out of total")
print((y_test != y_pred).sum, '/', ((y_test == y_pred).sum() + (y_test != y_pred).sum()))
print("percentage accuracy: ",100*accuracy_score(y_test,y_pred))