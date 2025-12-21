import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from log_exception import logging, CustomException
from sklearn.metrics import r2_score
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
                ("encoder", OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        
        logging.info("creating preprocessor")
        preprocessor = ColumnTransformer(
            [
                ("numeric_pipeline", num_pipelines, num_features),
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
    
    
def load_object(filepath):
    try:
        with open(filepath , "rb") as obj:
            return dill.load(obj)
    except Exception as e:
        raise CustomException(e, sys)


def params():
    try:
        params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                    'random_state': [0]
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256],
                    'random_state': [0]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256],
                    'random_state': [0]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    'random_state': [0]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                    'random_state': [0]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256],
                    'random_state': [0]
                }
                
            }
        return params
        
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            #y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            #train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    


def save_prediction(features_df: pd.DataFrame, prediction, file_path: str = "datasets/predictions.csv"):
    """
    Append input features with their predicted target to a CSV and print to terminal.

    - features_df: DataFrame containing the input features (one or more rows).
    - prediction: scalar or array-like predictions (one per row in features_df).
    - file_path: CSV file to append to (created if it doesn't exist).
    Returns: path to the CSV file.
    """
    try:
        preds = np.asarray(prediction)  # convert to numpy array
        # normalize to 1D array matching rows in features_df
        if preds.ndim == 0:
            preds = np.array([preds.item()])
        elif preds.ndim > 1:
            preds = preds.reshape(-1)

        df = features_df.copy().reset_index(drop=True)
        df["predicted_charges"] = preds.tolist()[: len(df)]
        df["prediction_time"] = datetime.now().isoformat()

        # ensure folder exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # write header if file doesn't exist, else append without header
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode="a", header=False, index=False)

        # print the saved rows to terminal
        print("Saved prediction(s):")
        print(df)

        return file_path
    except Exception as e:
        raise CustomException(e, sys)
    