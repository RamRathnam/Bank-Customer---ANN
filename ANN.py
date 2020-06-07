# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:24:32 2020

@author: arunr
"""

# Part 1 - Data Preprocessing

# Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv(r'C:\Users\arunr\Desktop\Data Science\Udemy\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 1 - Artificial Neural Networks (ANN)\Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

# Removing one column of dummy variable to avoid falling inot the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the keras libraries and packages
!pip install keras
!pip install tensorflow
import keras 
from keras.models import Sequential

# Initialising the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid')) 

# compiling the ANN ie., applying the stochastic gradient descent to the whole neural network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 3 - Making the predictions and evaluating the model
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_TorF = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_TorF)


# Accuracy of the model = (1542 + 146)/2000 = 0.844 ie., 84.40% accurate