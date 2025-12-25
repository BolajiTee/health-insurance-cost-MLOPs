import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from log_exception import logging, CustomException
from load_transformation import load_data, transform_data, save_preprocessor
from model_trainer import train_model
from prediction_pipeline import PredictPipeline, NewData
from utils import save_prediction
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
        # Validate and convert form inputs to proper types
        age_raw = request.form.get("age")
        sex = request.form.get("sex")
        bmi_raw = request.form.get("bmi")
        children_raw = request.form.get("children")
        smoker = request.form.get("smoker")
        region = request.form.get("region")

        try:
            age = int(age_raw)
        except (TypeError, ValueError):
            return render_template('home.html', error="Invalid age value")

        try:
            bmi = float(bmi_raw)
        except (TypeError, ValueError):
            return render_template('home.html', error="Invalid BMI value")

        try:
            children = int(children_raw)
        except (TypeError, ValueError):
            return render_template('home.html', error="Invalid children value")

        data = NewData(
            age=age,
            sex=sex,
            bmi=bmi,
            children=children,
            smoker=smoker,
            region=region,
        )
        logging.info("Data gathered from form")
        pred_df = data.make_feature_a_data_frame()
        print(pred_df)
        
        logging.info("Creating prediction pipeline object")
        pred_new_data = PredictPipeline()
        
        logging.info("Starting prediction.........")
        Charges = pred_new_data.prediction(pred_df)
        logging.info("Prediction completed")
        
        logging.info("Save and print the input + prediction to CSV and terminal")
        save_prediction(pred_df, Charges)
        
        return render_template("home.html", Charges = f"{Charges[0]:.2f}")









if __name__ == "__main__":
    # Run training pipeline before starting Flask server
    logging.info("Starting training pipeline")
    load = load_data()
    train_arr, test_arr, preprocessor = transform_data()
    save_preprocess = save_preprocessor(preprocessor)
    model_trainnig = train_model()
    logging.info("Training pipeline completed. Starting Flask server.")
    
    app.run(host= "0.0.0.0")