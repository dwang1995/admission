# ============================================================================ line 1 ============================================================================
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

#mode = "NN"
mode = "LR"

np.random.seed(25)

my_data = genfromtxt('Admission_Predict.csv', delimiter=',')

x = np.delete(my_data, 0, 0)
y = x[:, -1]
x = np.delete(x, 0, axis = 1)
x = np.delete(x, -1, axis = 1)

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20)

scalerX = MinMaxScaler(feature_range=(0, 1))
print(x_train[1])
x_train = scalerX.fit_transform(x_train)
print(x_train[1])
x_test = scalerX.transform(x_test)

if mode == "LR":
	lr = LinearRegression()
	lr.fit(x_train,y_train)
	y_head_lr = lr.predict(x_test)
	print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(x_test)[1]))
	print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test)[2]))
	print("real value of y_test[4]: " + str(y_test[4]) + " -> the predict: " + str(lr.predict(x_test)[4]))

	print("r_square score: ", r2_score(y_test,y_head_lr))
	print("mean square error: ", mean_squared_error(y_test,y_head_lr))

	y_head_lr_train = lr.predict(x_train)
	print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))
	print("mean square error for training dataset: ", mean_squared_error(y_train,y_head_lr_train))

if mode == "NN":
	model = models.Sequential()
	keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=10)
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
