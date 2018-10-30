# Multiple Linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap (deletes first column)
X = X[:, 1:]
# THis is taken care of in the background
# But sometimes have to do it manually, it's a good practice


# Splitting the dataset into the Training set and Test set
# Try putting 10 observations in test set and 40 observations in training set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# sklearn takes care of feature scaling in the background
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Fit the regressor to the training set
regressor.fit(X_train, y_train)
# Now the model will learn the correlations of the training set

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Preparation of backward elmination

# add a column of 1s to the (beginning 0f) independent data X 
# aooend function from numpy is used to add on data
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# Create new matrix of OPTIMAL features
# Contains only independent variables that are statistically significant to the dependent variable
# BACKWARD ELIMINATION STARTS HERE
# Create new regressor variable that utilizes statsmodels
# OLS: Ordinary Least Squares
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Use SUMMARY function to access P value
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Use SUMMARY function to access P value
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Use SUMMARY function to access P value
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Use SUMMARY function to access P value
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Use SUMMARY function to access P value
regressor_OLS.summary()

