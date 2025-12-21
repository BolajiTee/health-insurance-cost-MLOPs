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
        data = NewData(
            age = request.form.get("age"),
            sex = request.form.get("sex"),
            bmi = float(request.form.get("bmi")),
            children = request.form.get("children"),
            smoker = request.form.get("smoker"),
            region = request.form.get("region")
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
    
    app.run(host= "0.0.0.0", debug= True)