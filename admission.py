import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
from numpy import genfromtxt
from keras.utils import to_categorical
from keras import models
from keras import layers
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import operator
from sklearn.model_selection import KFold

#mode = "NN"
mode = "LR"

np.random.seed(25)

my_data = genfromtxt('Admission_Predict.csv', delimiter=',')

x = np.delete(my_data, 0, 0)
y = x[:, -1]
x = np.delete(x, 0, axis = 1)
x = np.delete(x, -1, axis = 1)

kf = KFold(n_splits=4)

for train_index, test_index in kf.split(x):
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

#x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20)



# temp = np.array([321,   99,     4,     3,     4  ,  7.325 ,  0  ])
# newArray = np.append( x_test, [temp], axis = 0 )
# print(newArray)


scalerX = MinMaxScaler(feature_range=(0, 1))
print(x_train[1])
x_train = scalerX.fit_transform(x_train)
print(x_train[1])
x_test = scalerX.transform(x_test)
# newArray = scalerX.fit_transform(newArray)
# print(newArray[-1])




def trainModel():
    dictionary = {}
    # NN
    nn = models.Sequential()
    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=10)  # weight random noramal to init meaight mean = 0.0
    # kernel_initializer='random_uniform', random_state = 20,
    nn.add(layers.Dense(8, activation="relu", input_shape=(len(x_train[0]),)))
    nn.add(layers.Dense(5, activation="relu"))
    nn.add(layers.Dense(1, activation="sigmoid"))
    # model.summary()

    nn.compile(optimizer="adam", loss="mean_squared_error")
    nn.fit(x_train, y_train, epochs=50, batch_size=5)  # , validation_data = (x_test, y_test))
    y_pred = nn.predict(x_test)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    mean_squared_error_nn = mean_squared_error(y_test, y_pred)
    print("========NN=============")
    print("r squared:", r2)
    print("mean_squared_error: ", mean_squared_error_nn)
    dictionary["NN"] = mean_squared_error_nn

    #linear regression

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    y_head_lr = lr.predict(x_test)


    print("========Linear Regression=============")
    r2_lr = r2_score(y_test, y_head_lr)
    mean_squared_error_lr = mean_squared_error(y_test, y_head_lr)
    print("r_square score: ", r2_lr)
    print("mean square error: ", mean_squared_error_lr)
    dictionary["LR"] = mean_squared_error_lr

    #decision tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(x_train, y_train)
    dtResult = dt.predict(x_test)
    print("========Decision Tree=============")
    print("r squared:", r2_score(y_test,dtResult))
    print("mean_squared_error: ", mean_squared_error(y_test, dtResult))
    #dictionary["DT"] = [r2_score(y_test,dtResult),mean_squared_error(y_test, dtResult)]
    dictionary["DT"] = mean_squared_error(y_test, dtResult)


    #random forrest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)


    rfResult = rf.predict(x_test)
    print("========Random Forrest=============")
    print("r_square score ", r2_score(y_test, rfResult))
    print("mean square error: ", mean_squared_error(y_test, rfResult))
    #dictionary["RF"] = [r2_score(y_test, rfResult),mean_squared_error(y_test, rfResult)]
    dictionary["RF"] = mean_squared_error(y_test, rfResult)


    print(dictionary)

    sorted_d = sorted(dictionary.items(), key=operator.itemgetter(1))
    print(sorted_d)
    return nn,lr,dt,rf


def getAdmissionProbability(gre_score, toefl_score, university_rating, sop, lor, college_gpa, research):

    nn,lr,dt,rf = trainModel()
    input = np.array([[gre_score, toefl_score, university_rating, sop, lor, college_gpa, research]])
    input = scalerX.transform(input)

    rf_pred = rf.predict(input)[-1]
    dt_pred = dt.predict(input)[-1]
    lr_pred = lr.predict(input)[-1]
    nn_pred = nn.predict(input)[-1]

    finalResult = 0.5*rf_pred + 0.3*dt_pred + 0.15*lr_pred + 0.05* nn_pred
    print("finalResult: ",finalResult)
    return(finalResult[0])

#getAdmissionProbability(337, 118, 4, 4.5, 4.5, 5, 1)
trainModel()
