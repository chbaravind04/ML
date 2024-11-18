# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:16:10 2024

@author: admin
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
data = load_iris()
iris_df = pd.DataFrame(data.data, columns=data.feature_names)
iris_df['class'] = data.target_names[data.target]

# Check for missing values
print("Missing Values:")
print(iris_df.isnull().sum())

# Replace missing values (if any)
# In this case, there are no missing values in the Iris dataset, so no replacement is needed.

# Split the data into training and testing sets
X = iris_df.drop('class', axis=1)
y = iris_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and test sets
print("\nShapes after splitting into Training and Test sets:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Split the dataset into k-folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Display the indices of the dataset in each fold
print("\nIndices in each fold:")
for train_index, test_index in kf.split(X):
    print("Train indices:", train_index)
    print("Test indices:", test_index)
    print("------")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the scaled features
print("\nScaled features:")
print("Scaled X_train:")
print(X_train_scaled)
print("\nScaled X_test:")
print(X_test_scaled)

# Plot using Seaborn pairplot for the scaled data
scaled_iris_df = pd.DataFrame(X_train_scaled, columns=data.feature_names)
scaled_iris_df['class'] = y_train.reset_index(drop=True)
sns.pairplot(scaled_iris_df, hue="class", diag_kind="kde")
plt.show()
from sklearn.preprocessing import OneHotEncoder

# Create an instance of OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the data
X_encoded = encoder.fit_transform(X)

# Get the feature names after one-hot encoding
feature_names = encoder.get_feature_names_out(input_features=X.columns)



