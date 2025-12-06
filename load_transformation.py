import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from log_exception import logging, CustomException

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
        raise CustomException(sys,e)
    
    
def transform_data():
    pass