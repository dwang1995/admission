import flask
from flask import Flask, Response, request, render_template, redirect, url_for,session
#import flask.ext.login as flask_login
import os, base64

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/input")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    #this is invoked when in the shell  you run
    #$ python app.py
    app.run(port=5000, debug=True)