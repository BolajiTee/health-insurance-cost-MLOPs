import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from log_exception import logging, CustomException
from utils import transformation_config, save_config

def load_data():
    try:
        logging.info("creating paths for raw, train and test data")
        raw_data_path: str = os.path.join("datasets", "raw_data.csv")
        train_data_path: str = os.path.join("datasets", "train_data.csv")
        test_data_path: str = os.path.join("datasets", "test_data.csv")
        
        logging.info("loading our dataset")
        with open("data/insurance_2.csv") as f:
            raw_data = pd.read_csv(f)
        
        logging.info("creating directory where different datasets would be stored ")    
        os.makedirs(os.path.dirname(raw_data_path),exist_ok=True)
        raw_data.to_csv(raw_data_path, index=False, header=True)
        
        logging.info("splitting train and test data")
        train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=123)
        train_data.to_csv(train_data_path, index=False, header=True)
        test_data.to_csv(test_data_path, index=False, header=True)
        
        logging.info("data loading is completed")
        
        return(
            train_data_path,
            test_data_path
        )
    except Exception as e:
        raise CustomException(e, sys)
    
    
def transform_data():
    try:
        logging.info("creating an instance preprocessor")
        preprocessor = transformation_config()
        
        logging.info("loading our train and test data before applying preprocessor on them")
        train_data = pd.read_csv("datasets/train_data.csv")
        test_data = pd.read_csv("datasets/test_data.csv")
        
        logging.info("separate features from target")
        target_column = "charges"
        
        
        train_features = train_data.drop(columns=[target_column], axis=1)
        train_target = train_data[target_column]
        
        test_features = test_data.drop(columns=[target_column], axis=1)
        test_target = test_data[target_column]
        
        logging.info("Applying preprocessor - Fit_transform on train and only transform on test")
        train_features = preprocessor.fit_transform(train_features)
        test_features = preprocessor.transform(test_features)
        
        logging.info("creating an array of our train and test data")
        train_arr = np.c_[
            train_features, np.array(train_target)
        ]
        
        test_arr = np.c_[
            test_features, np.array(test_target)
        ]
        
        return(
            train_arr, test_arr, preprocessor
        )
        
    except Exception as e:
        raise CustomException(e,sys)
    
    
def save_preprocessor(preprocessor):
    try:
        logging.info("creating path and pkl file of our preprocessor")
        preprocessor_path = os.path.join("datasets", "preprocessor.pkl")
        
        return (
            save_config(preprocessor_path, preprocessor)
        )  
    except Exception as e :
        raise CustomException(e,sys)