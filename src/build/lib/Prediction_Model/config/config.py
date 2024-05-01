import pathlib 
import os
import Prediction_Model
from datetime import datetime

"""General Paths"""
PACKAGE_ROOT = pathlib.Path(Prediction_Model.__file__).parent
APP_LOGS_DIR = os.path.join(PACKAGE_ROOT, "Application_Logs")
DATA_DIR = "DATA"
# Regex pattern for validating file names
FILE_NAME_PATTERN = r'^cement_strength_\d{%d}_\d{%d}\.csv$'

"""Training Paths"""
# Directories for storing validated training data
TRAINING_VALIDATED = "DATA/Raw_Training_Validated"
# Directory containing training batch files
TRAINING_FILES_DIR = os.path.join(PACKAGE_ROOT, DATA_DIR, "Training_Batch_Files")
# Directory for storing training data from the database
TRAINING_DATA_DIR = os.path.join(PACKAGE_ROOT, DATA_DIR, "Training_File_From_Db")
# Training data file name
TRAINING_DATA_FILE = "input_file.csv"
# Logs for training data ingestion
TRAINING_DATA_INGESTION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "train_data_ingestion_logs.log")
# Logs for training data validation
TRAINING_DATA_VALIDATION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "train_data_validation_logs.log")
# Logs for training process
TRAINING_LOGS = os.path.join(APP_LOGS_DIR, 'training_logs.log')
# Directories for good, bad, and archived training data
GOOD_RAW_DIR_TRAIN = os.path.join(PACKAGE_ROOT, TRAINING_VALIDATED, "Good_Raw_Data")
BAD_RAW_DIR_TRAIN = os.path.join(PACKAGE_ROOT, TRAINING_VALIDATED, "Bad_Raw_Data")
ARCHIVE_DIR_TRAIN = os.path.join(PACKAGE_ROOT, TRAINING_VALIDATED, "Archive")
# List of numeric columns in the training dataframe
NUMERIC_COLS = ['Cement _component_1', 'Blast Furnace Slag _component_2',
                'Fly Ash _component_3', 'Water_component_4',
                'Superplasticizer_component_5', 'Coarse Aggregate_component_6',
                'Fine Aggregate_component_7', 'Age_day',
                'Concrete_compressive _strength']

"""Prediction Paths"""
# Directories for storing validated prediction data
PREDICTION_VALIDATED = "DATA/Raw_Prediction_Validated"
# Directory containing prediction batch files
PREDICTION_FILES_DIR = os.path.join(PACKAGE_ROOT, DATA_DIR, "Prediction_Batch_Files")
# Directory for storing prediction data from the database
PREDICTION_DATA_DIR = os.path.join(PACKAGE_ROOT, DATA_DIR, "Prediction_File_From_Db")
# Prediction data file name
PREDICTION_DATA_FILE = 'prediction_input_file.csv'
# Logs for prediction data ingestion
PREDICTION_DATA_INGESTION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "prediction_data_ingestion_logs.log")
# Logs for prediction data validation
PREDICTION_DATA_VALIDATION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "prediction_data_validation_logs.log")
# Logs for training process
PREDICTION_LOGS = os.path.join(APP_LOGS_DIR, 'prediction_logs.log')
# Directories for good, bad, and archived prediction data
GOOD_RAW_DIR_PRED = os.path.join(PACKAGE_ROOT, PREDICTION_VALIDATED, "Good_Raw_Data")
BAD_RAW_DIR_PRED = os.path.join(PACKAGE_ROOT, PREDICTION_VALIDATED, "Bad_Raw_Data")
ARCHIVE_DIR_PRED = os.path.join(PACKAGE_ROOT, PREDICTION_VALIDATED, "Archive")
# List of numeric columns in the training dataframe
NUMERIC_COLS_PRED = ['Cement _component_1', 'Blast Furnace Slag _component_2',
                'Fly Ash _component_3', 'Water_component_4',
                'Superplasticizer_component_5', 'Coarse Aggregate_component_6',
                'Fine Aggregate_component_7', 'Age_day',
                ]
PREDICTION_OUTPUT_DIR = os.path.join(PACKAGE_ROOT,DATA_DIR,'Prediction_output')
PREDICTION_OUTPUT_FILE = os.path.join(PREDICTION_OUTPUT_DIR, 'prediction.csv')


"""Model Paths"""
# Directory for storing trained models
MODELS_DIR = os.path.join(PACKAGE_ROOT, "Models")
PREDICTION_MODELS_DIR = os.path.join(MODELS_DIR, "Prediction_Models")
# Random seed for reproducibility
RANDOM_SEED = 42
CLUSTERING_MODEL_NAME = 'Kmeans'

"""MLFlow and MySQL Configs"""
# URI for MLFlow tracking
MLFLOW_URI = 'http://localhost:5000'
# Table name for storing good raw data in the database
GOOD_RAW_TABLE_TRAIN = "good_training_data"
GOOD_RAW_TABLE_PREDICTION = "good_prediction_data"
# Chunk size for processing large datasets
CHUNK_SIZE = 20000
# MySQL Username and Password
USERNAME_MYSQL = os.environ.get('MYSQL_USER')
PASSWORD_MYSQL = os.environ.get('MYSQL_PASSWORD')
DATABASE = 'cement_strength_prediction'
HOST = 'localhost'
