# Polynomial Regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# 1:2 notation used for undexing to help python recognize column as array instead of vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Not needed because categorical data is not even being analuzed
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])"""


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting linear reggression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X, y)
'''lin_reg_predict = lin_reg.predict(X)'''


# Fitting multiple linear regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Create an instance of Polynomial Features only up to second degree (exponent)
# [p;y_reg is the variable that is being trained with given data
poly_reg = PolynomialFeatures(degree = 2)
# Fit the Modified X feature data regresion onto a new variable called X_poly
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


