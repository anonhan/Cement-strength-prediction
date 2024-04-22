import pandas as pd
import numpy as np
from Prediction_Model.config.config import PACAKAGE_ROOT, TRAINING_DATA_VALIDATION_LOGS_FILE, FILE_NAME_PATTERN, TRAINING_FILES_DIR
from Prediction_Model.data_ingestion.raw_data_validation import Raw_Data_Validation
from Prediction_Model.data_ingestion.data_loader import Data_Loader
from Prediction_Model.db_operations.db_operation import dBOperations
from Prediction_Model.app_logging.app_logger import App_Logger

class Train_Validation:
    def __init__(self, path):
        self.logger = App_Logger()
        self.raw_data_validator = Raw_Data_Validation(raw_files_path=path)
        self.db_operation_handler = dBOperations()
        self.logs_file = open(f"{TRAINING_DATA_VALIDATION_LOGS_FILE}", 'a+')
    
    def validate_training_data(self):
        """
        Description: Method to start validating the training data
        """
        try:
            self.logger.add_log(self.logs_file, "Started validating training data files...")
            LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName = self.raw_data_validator.get_values_from_schema()
            self.logger.add_log(self.logs_file, "Fetched JSON training schema details.")
            self.raw_data_validator.validate_file_name(FILE_NAME_PATTERN, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            self.logger.add_log(self.logs_file, "Validated training data file name format.")
            self.raw_data_validator.validate_column_names(NumberofColumns, ColName)
            self.logger.add_log(self.logs_file, "Validated training data file columns.")
            self.logger.add_log(self.logs_file, "Raw Data Validation Complete!!")

            self.logger.add_log(self.logs_file, "Started dB operations...")
            self.db_operation_handler.create_table(ColName)
            self.logger.add_log(self.logs_file, "Created table in dB.")
            self.db_operation_handler.insert_good_data_into_db()
            self.logger.add_log(self.logs_file, "Inserted Good Raw Data in dB.")
            self.raw_data_validator.move_bad_files_to_archive()
            self.logger.add_log(self.logs_file, "Moved Bad Data files to archive.")
            self.raw_data_validator.remove_good_bad_dirs()
            self.logger.add_log(self.logs_file, "Deleted Good and Bad Raw Data folders.")
            self.db_operation_handler.select_data_from_table()
            self.logger.add_log(self.logs_file, "Fetched training data from dB.")

            self.logger.add_log(self.logs_file, "All operations of Train_Validation are completed !!")

        except Exception as e:
            self.logger.add_log(self.logs_file, "Error while validating training data::"+str(e))
            self.logs_file.close()
            raise Exception()