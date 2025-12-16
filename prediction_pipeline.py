import os
import sys
import pandas as pd
from load_transformation import logging, CustomException
from utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        model_path = "datasets/best_model.pkl"
        preprocessor_path = "datasets/preprocessor.pkl"
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        scaled_data = preprocessor.transform(features)
        pred = model.predict(scaled_data)
        return pred
    
    
class NewData:
    def __init__(self,
                 age: int, 
                 sex: str, 
                 bmi: float, 
                 children: str, 
                 smoker: str, 
                 region: str):
        
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
    
    def make_feature_a_data_frame(self):
        try:
            new_data_dict = {
                "age" : [self.age],
                "sex" : [self.sex],
                "bmi" : [self.bmi],
                "children" : [self.children],
                "smoker" : [self.smoker],
                "region" : [self.region]
                }
            return pd.DataFrame(new_data_dict)
        except Exception as e:
            raise CustomException(e, sys)