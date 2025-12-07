import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from log_exception import logging, CustomException
import dill



def transformation_config():
    try:
        num_features = ['age', 'bmi']
        cat_features = ['sex', 'children', 'smoker', 'region']
        
        logging.info("creating numerical and categorical pipelines")
        num_pipelines = Pipeline(
            steps= [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        
        cat_pipelines = Pipeline(
            steps= [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder())
            ]
        )
        
        logging.info("creating preprocessor")
        preprocessor = ColumnTransformer(
            [
                ("numeric_pipeline", num_pipelines, num_features)
                ("categorical_pipeline", cat_pipelines, cat_features)
            ]
        )
    
        return preprocessor
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
def save_config(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open (file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    