# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:48:55 2018

@author: johnyorangeseed
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import and section the datasets
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values



# Splitting the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
# Immediately define the variables required for the training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0) 
# w/ test_size of 0.2 and 10 observations, 2 go to test & 8 go to training
# X corresponds to the feature variables,
# Y corresponds to the dependent variables
# random_state is the seed of random # generator, can be any int value

"""
# Feature Scaling
# import StandardScaler class for feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# fit and transform the sc_X(StandardScaler) object onto training variable (features)
X_train = sc_X.fit_transform(X_train)
# use sc_X to only transform corresponding test variable (features)
X_test = sc_X.transform(X_test)
"""

''' feature scaling not required in linear regression because library for simple linear regression will
take care of that for us '''

# Fitting simple linear regression to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# Now the model will learn the correlations of the training set once it is applied through the code above

y_pred = regressor.predict(X_test)
# Predictions are not always spot on because linear regression is being used. 

# Visualizing the Traning set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()













