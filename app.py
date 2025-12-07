import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from log_exception import logging, CustomException
from load_transformation import load_data, transform_data, save_preprocessor


if __name__ == "__main__":
    load = load_data()
    transform = transform_data()
    save_preprocess = save_preprocessor()