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

#mode = "NN"
mode = "LR"

np.random.seed(25)

my_data = genfromtxt('Admission_Predict.csv', delimiter=',')

x = np.delete(my_data, 0, 0)
y = x[:, -1]
x = np.delete(x, 0, axis = 1)
x = np.delete(x, -1, axis = 1)

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20)

print(x_test[1])
print(x_test[2])
print(x_test[3])

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


def main():
    if mode == "LR":
        lr = LinearRegression()
        lr.fit(x_train,y_train)



        y_head_lr = lr.predict(x_test)
        print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(x_test)[1]))
        print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test)[2]))
        print("real value of y_test[4]: " + str(y_test[4]) + " -> the predict: " + str(lr.predict(x_test)[4]))

        # print( "shizhan's chance : " + str(lr.predict(newArray)[-1]) )

        print("r_square score: ", r2_score(y_test,y_head_lr))
        print("mean square error: ", mean_squared_error(y_test,y_head_lr))

        y_head_lr_train = lr.predict(x_train)
        print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))
        print("mean square error for training dataset: ", mean_squared_error(y_train,y_head_lr_train))

        # print("Your chance is : " + str(lr.predict(newArray)[-1]))

    if mode == "NN":
        model = models.Sequential()
        keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=10) #weight random noramal to init meaight mean = 0.0
        # kernel_initializer='random_uniform', random_state = 20,
        model.add(layers.Dense(8, activation = "relu", input_shape=(len(x_train[0]), )))
        model.add(layers.Dense(5, activation = "relu"))
        model.add(layers.Dense(1, activation = "sigmoid"))
        model.summary()

        model.compile(optimizer = "adam", loss = "mean_squared_error")
        results = model.fit(x_train, y_train, epochs= 50, batch_size = 5)#, validation_data = (x_test, y_test))
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        print("r squared:", r2)
        print("mean_squared_error: ", mean_squared_error(y_test, y_pred))


def trainModel():
    dictionary = {}
    # NN
    model = models.Sequential()
    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=10)  # weight random noramal to init meaight mean = 0.0
    # kernel_initializer='random_uniform', random_state = 20,
    model.add(layers.Dense(8, activation="relu", input_shape=(len(x_train[0]),)))
    model.add(layers.Dense(5, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # model.summary()

    model.compile(optimizer="adam", loss="mean_squared_error")
    results = model.fit(x_train, y_train, epochs=50, batch_size=5)  # , validation_data = (x_test, y_test))
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    mean_squared_error_nn = mean_squared_error(y_test, y_pred)
    print("========NN=============")
    print("r squared:", r2)
    print("mean_squared_error: ", mean_squared_error_nn)
    dictionary["NN"] = [r2,mean_squared_error_nn]

    #linear regression

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    y_head_lr = lr.predict(x_test)


    print("========Linear Regression=============")
    r2_lr = r2_score(y_test, y_head_lr)
    mean_squared_error_lr = mean_squared_error(y_test, y_head_lr)
    print("r_square score: ", r2_lr)
    print("mean square error: ", mean_squared_error_lr)
    dictionary["LR"] = [r2_lr,mean_squared_error_lr]

    #decision tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(x_train, y_train)
    dtResult = dt.predict(x_test)
    print("========Decision Tree=============")
    print("r squared:", r2_score(y_test,dtResult))
    print("mean_squared_error: ", mean_squared_error(y_test, dtResult))
    dictionary["DT"] = [r2_score(y_test,dtResult),mean_squared_error(y_test, dtResult)]


    #random forrest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)


    rfResult = rf.predict(x_test)
    print("========Random Forrest=============")
    print("r_square score ", r2_score(y_test, rfResult))
    print("mean square error: ", mean_squared_error(y_test, rfResult))
    dictionary["RF"] = [r2_score(y_test, rfResult),mean_squared_error(y_test, rfResult)]


    print(dictionary)




trainModel()


