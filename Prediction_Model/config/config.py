import pathlib 
import os
import Prediction_Model
from datetime import datetime

PACAKAGE_ROOT = pathlib.Path(Prediction_Model.__file__).parent

# Logs Directory
APP_LOGS_DIR = os.path.join(PACAKAGE_ROOT, "Application_Logs")

# Logs file creation
# DATA_INGESTION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "Data_Ingesion_Logs/data_ingestion_logs.log")
DATA_INGESTION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "data_ingestion_logs.log")
TRAINING_DATA_VALIDATION_LOGS_FILE = os.path.join(APP_LOGS_DIR, "train_data_validation_logs.log")
TRAINING_LOGS = os.path.join(APP_LOGS_DIR, 'training_logs.log')

# Good, bad and archive data Directory
DATA_DIR = "DATA"
TRAINING_VALIDATED = "DATA/Raw_Training_Validated"
GOOD_RAW_DIR = os.path.join(PACAKAGE_ROOT, TRAINING_VALIDATED, "Good_Raw_Data")
BAD_RAW_DIR = os.path.join(PACAKAGE_ROOT, TRAINING_VALIDATED, "Bad_Raw_Data")
ARCHIVE_DIR = os.path.join(PACAKAGE_ROOT, TRAINING_VALIDATED, "Archive")

# Trainig Data Directory
TRAINING_FILES_DIR = os.path.join(PACAKAGE_ROOT, DATA_DIR, "Training_Batch_Files")

# File name validation Regex
FILE_NAME_PATTERN = r'^cement_strength_\d{%d}_\d{%d}\.csv$'
GOOD_RAW_TABLE = "good_raw_data"

# Final training file directory
TRAINING_DATA_DIR = os.path.join(PACAKAGE_ROOT, DATA_DIR, "Training_File_From_Db")
TRAINING_DATA_FILE = "input_file.csv"
CHUNK_SIZE = 200

# Trained Models directory
MODELS_DIR = os.path.join(PACAKAGE_ROOT, "Models")
RANDOM_SEED = 42

# Numeric columns in the dataframe
NUMERIC_COLS = ['Cement _component_1', 'Blast Furnace Slag _component_2',
                'Fly Ash _component_3', 'Water_component_4',
                'Superplasticizer_component_5', 'Coarse Aggregate_component_6',
                'Fine Aggregate_component_7', 'Age_day',
                'Concrete_compressive _strength']
# MLFlow tracking URI
MLFLOW_URI = 'http://localhost:5000'