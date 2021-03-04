# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 03:02:17 2019

@author: Mohit
"""
#importing various libraries to be used 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing, cross_validation, neighbors
from firebase import firebase

#Reading csv file
dataset = pd.read_csv('Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

#Filling Missing values
imputer = Imputer(missing_values='NaN',strategy = 'mean', axis=0)
imputer = imputer.fit(x[:, 1:6])
x[:, 1:6]=imputer.transform(x[:, 1:6])

#Changing alphabetic values to numeric/binary format
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y =labelencoder_y.fit_transform(y)

#Calculating the accracy of Model
#acc=0
#for i in range(10):
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.254)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train,y_train)
    accuracy = clf.score(x_test, y_test)
 #   acc+=accuracy
print(accuracy)

#Establishing connection with firebase
firebase = firebase.FirebaseApplication("https://testproject-3cd5a.firebaseio.com/",None)
#Retreiving values from firebase via json object result in the form of dictionary
result = firebase.get("/User1",None)
print(result)
#fetching the value of gender from json object
gen = result['Gender']
m = 1
f = 0
if(gen=='M'):
    m=1
else:
    f=1
    m=0
    
#Setting all values in required trained format in order to test and predict the goal
example = np.array([m,f,result['Age'],result['Typing'],result['Scrolling'],result['Backspaces'],result['Timegap']])
example = example.reshape(1,-1)
prediction = clf.predict(example)
if(prediction==1):
    print("Hey! Looking in some Problem.. How can I help you?")
else:
    print("You are in a good mood")
