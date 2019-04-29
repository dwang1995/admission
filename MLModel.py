import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def getAdmissionProbability(gre_score, tofel_score, university_rating, sop, lor, college_gpa, research):

	np.random.seed(25)

	my_data = genfromtxt('Admission_Predict.csv', delimiter=',')

	x = np.delete(my_data, 0, 0)
	y = x[:, -1]
	x = np.delete(x, 0, axis = 1)
	x = np.delete(x, -1, axis = 1)

	x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20)

	temp = np.array([gre_score,tofel_score,university_rating,sop,lor,college_gpa,research])
	x_test = np.append( x_test, [temp], axis = 0 )

	temp = np.array([gre_score,   tofel_score,     university_rating,     sop,     lor  ,  college_gpa ,  research  ])
	newArray = np.append( x_test, [temp], axis = 0 )

	scalerX = MinMaxScaler(feature_range=(0, 1))
	x_train = scalerX.fit_transform(x_train)
	x_test = scalerX.transform(x_test)
	newArray = scalerX.fit_transform(newArray)

	lr = LinearRegression()
	lr.fit(x_train,y_train)

	y_head_lr = lr.predict(x_test)

	return lr.predict(newArray)[-1] 

print( getAdmissionProbability(337, 118, 4, 4.5, 4.5, 9.65, 1) )
print( getAdmissionProbability(329, 100, 4, 4, 4, 9, 0) )
print( getAdmissionProbability(334, 110, 5, 4.5, 4.5, 9.95, 1) )

