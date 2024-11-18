# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:26:34 2024

@author: zone
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# Load the Iris dataset
data = pd.read_csv("E:/deep learning/Deeplearning_lab/Iris.csv")

# Select features and target
X = data.drop("Species", axis=1)
y = data['Species']
# Encoding the Species column to get numerical class
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier on the training data
gnb.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = gnb.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy of Prediction on Iris Flower is: {accuracy}")

