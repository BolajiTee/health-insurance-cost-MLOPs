import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from log_exception import logging, CustomException
from load_transformation import load_data, transform_data, save_preprocessor
from model_trainer import train_model
from prediction_pipeline import PredictPipeline, NewData
from flask import Flask, request, render_template

application = Flask(__name__)

app = application

@app.route('/')
def index():
   return render_template('index.html') 

@app.route('/predict_data', methods = ['GET', 'POST'])
def new_predict(): # this "new_predict" will be in the form in the templates
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = NewData(
            age = request.form.get("age"),
            sex = request.form.get("sex"),
            bmi = request.form.get("bmi"),
            children = request.form.get("children"),
            smoker = request.form.get("smoker"),
            region = request.form.get("region")
        )
        
        pred_df = data.make_feature_a_data_frame()
        print(pred_df)
        pred_new_data = PredictPipeline()
        new_target = pred_new_data.predict(pred_df)
        
        return render_template("home.html", new_target = new_target[0])









if __name__ == "__main__":
    app.run(host= "0.0.0.0", debug= True)
    load = load_data()
    transform = transform_data()
    save_preprocess = save_preprocessor()
    model_trainnig = train_model()