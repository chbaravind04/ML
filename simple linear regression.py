# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 06:00:23 2024

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('D:\machine learning\Complete-Machine-Learning-2023-main\height-weight.csv')
df.head()
##scatter plot
plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
## Correlation
df.corr()
## Seaborn for visualization
import seaborn as sns
sns.pairplot(df)
## Independent and dependent features
X=df[['Weight']] ### independent features should be data frame or 2 dimesnionalarray
y=df['Height'] ## this variiable can be in series or 1d array
X_series=df['Weight']
np.array(X_series).shape
np.array(y).shape
## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
## Standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
## Apply Simple Linear Regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression(n_jobs=-1)
regression.fit(X_train,y_train)
print("Coefficient or slope:",regression.coef_)
print("Intercept:",regression.intercept_)
## plot Training data plot best fit line
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))
y_pred=regression.predict(X_test)
## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)