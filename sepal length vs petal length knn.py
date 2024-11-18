# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:57:31 2024

@author: zone
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:04:22 2024

@author: zone
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

# Load the Iris dataset
dataset = pd.read_csv("E:/deep learning/Deeplearning_lab/Iris.csv")
print(dataset.shape)
print(dataset.head())
print(dataset.tail(2))
print(dataset.describe())

# Extract features (X) and target labels (y)
X = dataset.iloc[:, 1:5].values  # Features (sepal length, sepal width, petal length, petal width)
y = dataset.iloc[:, 5].values    # Target (species)

# Convert species labels to numeric values using LabelEncoder
le = LabelEncoder()
y_numeric = le.fit_transform(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.20, random_state=42)

# Train the model using KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train, y_train)

# Predict using the classifier on the test data
y_pred = classifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print the accuracy score
accuracy = accuracy_score(y_test, y_pred) * 100
print('Accuracy of our model is equal to ' + str(round(accuracy, 2)) + ' %.')

# Plotting Training and Testing Data on the same plot
plt.figure(figsize=(8, 6))

# Scatter plot for training data
plt.scatter(X_train[:, 0], X_train[:, 2], c=y_train, cmap='viridis', marker='o', label='Training data')

# Scatter plot for test data with predicted labels
plt.scatter(X_test[:, 0], X_test[:, 2], c=y_pred, cmap='coolwarm', marker='x', label='Test data (predicted)', edgecolor='k')

# Adding title and labels
plt.title('Training and Test Data (Sepal Length vs Petal Length)')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')

# Add colorbars and legend
plt.colorbar(label='Species')
plt.legend()

# Grid and layout
plt.grid(True)
plt.tight_layout()
plt.show()
