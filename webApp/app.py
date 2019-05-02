import flask
from flask import Flask, Response, request, render_template, redirect, url_for,session
#import flask.ext.login as flask_login
import os, base64
from decimal import Decimal


import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.api as sm

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('input.html')

@app.route("/input",methods=['POST'])
def calculatePercentage():

    gre = int(request.form.get("GRE"),10)
    toefl = int(request.form.get("TOEFL"),10)
    rate = int(request.form.get("rate"),10)
    sop = Decimal(request.form.get("SOP"))
    lor = Decimal(request.form.get("LOR"))
    gpa = Decimal(request.form.get("GPA"))
    research = request.form.get("ResearchCheck")
    researchReturn = ""
    if(research == None ):
        research = 0;
        researchReturn = "No"
    else:
        research = 1
        researchReturn = "Yes"


    #print(gre,toefl,rate,sop,lor,gpa,research)
    print(request.form.get("ResearchCheck"))
    result = getAdmissionProbability(gre,toefl,rate,sop,lor,gpa,research)
    returnList = [gre,toefl,rate,sop,lor,gpa,researchReturn,result]

    return render_template("result.html",results=returnList)





def getAdmissionProbability(gre_score, tofel_score, university_rating, sop, lor, college_gpa, research):
    np.random.seed(25)

    my_data = genfromtxt('Admission_Predict.csv', delimiter=',')

    x = np.delete(my_data, 0, 0)
    y = x[:, -1]
    x = np.delete(x, 0, axis=1)
    x = np.delete(x, -1, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    scalerX = MinMaxScaler(feature_range=(0, 1))
    x_train = scalerX.fit_transform(x_train)
    x_test = scalerX.transform(x_test)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    model = sm.OLS(y_train,x_train)
    result = model.fit()
    print(result.summary())



    x_input = np.array([[gre_score, tofel_score, university_rating, sop, lor, college_gpa, research]])
    x_input = scalerX.transform(x_input)
    #print("coef ",lr.coef_)

    return lr.predict(x_input)[-1]

if __name__ == "__main__":
    #this is invoked when in the shell  you run
    #$ python app.py
    app.run(port=5000, debug=True)