# -*- coding: utf-8 -*-

"""
Created on Thu Jan 11 06:43:15 2024

@author: admin
"""
#Comapre two methods of linear regression 
import numpy as np
import matplotlib.pyplot as plt
# simple linear regression using solving set of equations and to find the co-efficients directly
# Generate the synthetic data
x = np.linspace(1, 15, 100)
y = 2*x + (x + np.random.rand(len(x)))**2
# suppose if your x and y values are given in tablur format then instead of simulated data replace those values with x and y 
# Linear Regression
X_linear = np.vstack((np.ones_like(x), x))
b_linear = np.linalg.pinv(X_linear.T) @ y
plt.figure(1)
plt.scatter(x, y)
plt.plot(x, X_linear.T @ b_linear, color='r')
plt.title('Linear Regression')
print("Coefficients of Linear Regression:", b_linear)

# Second-degree Polynomial Regression
X_poly2 = np.vstack((np.ones_like(x), x, x**2))
b_poly2 = np.linalg.pinv(X_poly2.T) @ y
plt.figure(2)
plt.scatter(x, y)
plt.plot(x, X_poly2.T @ b_poly2, color='r')
plt.title('Second-degree Polynomial Regression')
plt.show()
print("Coefficients of Second-degree Polynomial Regression:", b_poly2)
intercept_poly2 = b_poly2[0]
print("Intercept of Second-degree Polynomial Regression:", intercept_poly2)
# simple linear regression using solving set of equations and to find the co-efficients using LR ML model for python 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# PolynomialFeatures class in scikit-learn is used for feature engineering in polynomial regression. 
# It generates polynomial features based on the input data, allowing you to capture nonlinear relationships in your model
# Given data
x = np.linspace(1, 15, 100)
y = 2 * x + (x + np.random.rand(len(x)))**2

# Linear Regression
X_linear = x.reshape(-1, 1)
model_linear = LinearRegression().fit(X_linear, y)
print("Slope:", model_linear.coef_[0])
print("Intercept:", model_linear.intercept_)
plt.figure(1)
plt.scatter(x, y)
plt.plot(x, model_linear.predict(X_linear), color='r')
plt.title('Linear Regression')


# Second-degree Polynomial Regression
X_poly2 = x.reshape(-1, 1)
model_poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
model_poly2.fit(X_poly2, y)

# Accessing coefficients
linear_regression_coef = model_poly2.steps[-1][1].coef_
intercept = model_poly2.steps[-1][1].intercept_
print("Coefficients of Second-degree Polynomial Regression:")
print("Coefficients:", linear_regression_coef)
print("Intercept:", intercept)

plt.figure(2)
plt.scatter(x, y)
plt.plot(x, model_poly2.predict(X_poly2), color='r')
plt.title('Second-degree Polynomial Regression')
plt.show()
