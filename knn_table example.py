# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:44:55 2024

@author: zone
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('E:/machine learning/knearest.csv')
print(dataset.shape)
print(dataset.head())
print(dataset.tail(2))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
predicted = classifier.predict([[2,20,3.1]])
print(predicted)