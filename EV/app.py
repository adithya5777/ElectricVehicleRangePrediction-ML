from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    values = [float(x) for x in request.form.values()]
    final_features = [np.array(values)]
    print(final_features)
    arr=preprocessing.normalize(final_features)
    prediction = model.predict(arr)


    new = np.round(prediction, 2)

    return render_template('result.html', a1=new[0][0], a2=new[0][1],a3=new[0][2],a4=new[0][3],a5=new[0][4],a6=new[0][5],a7=new[0][6])
    

if __name__=="__main__":
    app.run(debug=True, port=8000)
