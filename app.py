from flask import Flask, request, render_template
import pickle
import time
import pandas as pd
import jinja2
from pycaret.regression import *
import numpy as np
import json

app = Flask(__name__)

def parseInt(x):
    return int(x) if x.isdigit() else 0


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/test', methods=['GET'])
def test():
    a = pd.read_csv('test.csv')
    return render_template('test.html', model=a)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/getResponseLinearReg', methods=["GET", "POST"])
def getResponseLinearReg():
    time.sleep(1)

    CLEARSKYDHI = parseInt(request.form["CLEARSKYDHI"])
    CLEARSKYGHI = parseInt(request.form["CLEARSKYGHI"])
    DEWPOINT = parseInt(request.form["DEWPOINT"])
    DHI = parseInt(request.form["DHI"])
    GHI = parseInt(request.form["GHI"])
    OZONE = parseInt(request.form["OZONE"])
    RELATIVEHUMADITY = parseInt(request.form["RELATIVEHUMADITY"])
    SOLARZENITHANGLE = parseInt(request.form["SOLARZENITHANGLE"])
    SURFACEALBEDO = parseInt(request.form["SURFACEALBEDO"])
    PRECIPITABLEWATER = parseInt(request.form["PRECIPITABLEWATER"])
    CLOUDTYPE = parseInt(request.form["CLOUDTYPE"])

    CLOUDTYPE_ARR = [0]*10

    if CLOUDTYPE in range(0,10):
        CLOUDTYPE_ARR[CLOUDTYPE] = 1

    del CLOUDTYPE_ARR[5]



    inputList = [CLEARSKYDHI,
                 CLEARSKYGHI,
                 DEWPOINT,
                 DHI,
                 GHI,
                 OZONE,
                 RELATIVEHUMADITY,
                 SOLARZENITHANGLE,
                 SURFACEALBEDO,
                 
                 PRECIPITABLEWATER] + CLOUDTYPE_ARR

    # inputList = [0] * 27

    # pickled_model = pickle.load(open('trainedmodelasd11', 'rb'))

    # return str(pickled_model.predict([inputList],))
    with open("Lastversion", 'rb') as file:
        pickle_model = pickle.load(file)
        y_pred_from_pkl = pickle_model.predict([inputList])
    # print(y_pred_from_pkl)
    return str(y_pred_from_pkl[0])


if __name__ == '__main__':
    app.run(debug=True)
