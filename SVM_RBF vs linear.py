# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:56:29 2024

@author: zone
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the Iris dataset
df = pd.read_csv("E:/machine learning/ML_lab data/iris.csv")

# Display first 10 rows
print(df.head(10))

# Extract features and labels (use only the first two features for visualization)
X = df.iloc[:, [0, 1]].values  # Sepal length and sepal width
y = df.iloc[:, 4].values  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create an SVM model with a linear kernel
model_linear = SVC(kernel='linear', random_state=0)
model_linear.fit(X_train, y_train)
pred_linear = model_linear.predict(X_test)

# Confusion matrix and accuracy for linear kernel
print("\nConfusion matrix for linear kernel:\n", confusion_matrix(y_test, pred_linear))
print("\nAccuracy for linear kernel:", accuracy_score(y_test, pred_linear))
print("\nClassification report for linear kernel:\n", classification_report(y_test, pred_linear))

# Create an SVM model with RBF kernel
model_rbf = SVC(kernel='rbf', random_state=0)
model_rbf.fit(X_train, y_train)
pred_rbf = model_rbf.predict(X_test)

# Confusion matrix and accuracy for RBF kernel
print("\nConfusion matrix for RBF kernel:\n", confusion_matrix(y_test, pred_rbf))
print("\nAccuracy for RBF kernel:", accuracy_score(y_test, pred_rbf))
print("\nClassification report for RBF kernel:\n", classification_report(y_test, pred_rbf))

# Function to plot decision boundaries
def plot_decision_boundary(X, y, model, title):
    # Set up meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Predict on each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(('red', 'green', 'blue')))
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(title)
    plt.show()

# Plot decision boundaries for linear and RBF kernels
plot_decision_boundary(X, y, model_linear, "SVM Decision Boundary with Linear Kernel")
plot_decision_boundary(X, y, model_rbf, "SVM Decision Boundary with RBF Kernel")
