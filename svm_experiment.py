# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 06:27:38 2024

@author: admin
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Reading the csv Iris dataset file
df = pd.read_csv("E:/machine learning/ML_lab data/iris.csv")
print(df.head(10))
# Conditioning the data
array = df.values
X = array[:,0:4]
y = array[:,4]
# Condition the training and Testing data
# The number of samples can be tuned with the test_size parameter.
# Here, 95% of the data is used.
X_train, X_test, y_train, y_test = train_test_split( \
 X, y,test_size=0.95,random_state=0)
# Construct the Linear model
model = SVC(kernel='linear',random_state=0)
model.fit(X_train,y_train)
# Test the model with Linear kernel
pred = model.predict(X_test)
# Prepare confusion matrix
print("\n\nThe confusion matrix is \n\n")
conf = confusion_matrix(y_test,pred)
print(conf)
# pepare Classification Report
print("\n\nAccuracy is")
accuracy = accuracy_score(y_test,pred)
print(accuracy)
# Or report can be obtained as follows
print('\n Classification Report')
report = classification_report(y_test,pred)
print(report)
# RBF kernel
model1 = SVC(kernel='rbf',random_state=0)
model1.fit(X_train,y_train)
# Test the model
pred = model1.predict(X_test)
# Prepare confusion matrix
print("\n\nThe confusion matrix for RBF kernel is \n\n")
conf = confusion_matrix(y_test,pred)
print(conf)
# pepare Classification Report
print("\n\nAccuracy for RBF kernel is")
accuracy = accuracy_score(y_test,pred)
print(accuracy)

