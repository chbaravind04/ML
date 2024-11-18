# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:47:17 2024

@author: admin
"""

import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
# Read the dataset
df = pd.read_csv("E:/machine learning/ML_lab data/iris.csv")
df.columns = ['x1', 'x2', 'x3', 'x4', 'y']

# Drop unnecessary columns
df = df.drop(['x3', 'x4'], axis=1)

# Print the first 10 rows of the DataFrame
print(df.head(10))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
X = df.values[:, 0:2]
kmeans.fit(X)
df['pred'] = kmeans.predict(X)

# Plot the clusters
plt.scatter('x1', 'x2', c=df['pred'] ,edgecolors='red')
# plt.scatter(X_test[:,0], X_test[:,1], c=df['pred'] , marker= '*', s=100,edgecolors='red')


df = pd.read_csv("iris.csv")
df.columns =['x1','x2','x3','x4','y']
df = df.drop(['x3','x4'],1)
# print(df.head(10))
kmeans = KMeans(n_clusters = 3)
X=df.values[:,0:2]
kmeans.fit(X)
df['pred'] = kmeans.predict(X)
df.head(10)
sns.lmplot('x1','x2',scatter=True,fit_reg=False,data=df,hue='pred')