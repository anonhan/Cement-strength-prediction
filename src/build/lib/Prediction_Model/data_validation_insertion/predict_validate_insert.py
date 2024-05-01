from Prediction_Model.config.config import (PREDICTION_DATA_VALIDATION_LOGS_FILE, 
                                            FILE_NAME_PATTERN,
                                            GOOD_RAW_DIR_PRED,
                                            BAD_RAW_DIR_PRED,
                                            ARCHIVE_DIR_PRED,
                                            PREDICTION_DATA_INGESTION_LOGS_FILE,
                                            PREDICTION_DATA_DIR,
                                            PREDICTION_DATA_FILE,
                                            PREDICTION_FILES_DIR,
                                            GOOD_RAW_TABLE_PREDICTION)
from Prediction_Model.data_ingestion.raw_data_validation import Raw_Data_Validation
from Prediction_Model.db_operations.db_operation import dBOperations
from Prediction_Model.app_logging.app_logger import App_Logger

class Prediction_Validation:
    def __init__(self, path):
        """
        Initialize Prediction_Validation object.

        Parameters:
        path (str): Path to the directory containing prediction data files.
        """
        self.logger = App_Logger()
        self.raw_data_validator = Raw_Data_Validation(raw_files_path=path, json_schema_file='schema_prediction.json')
        self.db_operation_handler = dBOperations()
        self.logs_file = open(PREDICTION_DATA_VALIDATION_LOGS_FILE, 'a+')

    def validate_prediction_data(self, is_file_from_path=True, uploaded_file=None, file_name=None):
        """
        Validate the prediction data files.
        """
        try:
            self.logger.add_log(self.logs_file, "Started validating prediction data files...")
            LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName = self.raw_data_validator.get_values_from_schema(PREDICTION_DATA_INGESTION_LOGS_FILE)
            self.logger.add_log(self.logs_file, "Fetched JSON prediction schema details.")
            if is_file_from_path:
                self.raw_data_validator.validate_file_name(FILE_NAME_PATTERN, 
                                                        LengthOfDateStampInFile, 
                                                        LengthOfTimeStampInFile, 
                                                        PREDICTION_DATA_INGESTION_LOGS_FILE, 
                                                        GOOD_RAW_DIR_PRED, 
                                                        BAD_RAW_DIR_PRED, 
                                                        ARCHIVE_DIR_PRED,
                                                        PREDICTION_FILES_DIR)
            else:
                self.raw_data_validator.validate_file_name_without_path(FILE_NAME_PATTERN, 
                                            LengthOfDateStampInFile, 
                                            LengthOfTimeStampInFile, 
                                            PREDICTION_DATA_INGESTION_LOGS_FILE, 
                                            GOOD_RAW_DIR_PRED, 
                                            BAD_RAW_DIR_PRED, 
                                            ARCHIVE_DIR_PRED,
                                            uploaded_file,
                                            file_name)
            self.logger.add_log(self.logs_file, "Validated prediction data file name format.")
            self.raw_data_validator.validate_column_names(NumberofColumns, 
                                                          ColName, 
                                                          PREDICTION_DATA_INGESTION_LOGS_FILE,
                                                          GOOD_RAW_DIR_PRED, 
                                                          BAD_RAW_DIR_PRED,
                                                          ARCHIVE_DIR_PRED)
            self.logger.add_log(self.logs_file, "Validated prediction data file columns.")
            self.logger.add_log(self.logs_file, "Raw Data Validation Complete!!")

            self.logger.add_log(self.logs_file, "Started dB operations...")
            self.db_operation_handler.create_table(ColName, PREDICTION_DATA_INGESTION_LOGS_FILE, GOOD_RAW_TABLE_PREDICTION)
            self.logger.add_log(self.logs_file, "Created table in dB.")
            self.db_operation_handler.insert_good_data_into_db(PREDICTION_DATA_INGESTION_LOGS_FILE, GOOD_RAW_DIR_PRED, GOOD_RAW_TABLE_PREDICTION)
            self.logger.add_log(self.logs_file, "Inserted Good Raw Data in dB.")
            self.raw_data_validator.move_bad_files_to_archive(PREDICTION_DATA_INGESTION_LOGS_FILE, BAD_RAW_DIR_PRED, ARCHIVE_DIR_PRED)
            self.logger.add_log(self.logs_file, "Moved Bad Data files to archive.")
            self.raw_data_validator.remove_good_bad_dirs(PREDICTION_DATA_INGESTION_LOGS_FILE, GOOD_RAW_DIR_PRED, BAD_RAW_DIR_PRED)
            self.logger.add_log(self.logs_file, "Deleted Good and Bad Raw Data folders.")
            self.db_operation_handler.select_data_from_table(PREDICTION_DATA_INGESTION_LOGS_FILE, PREDICTION_DATA_DIR, PREDICTION_DATA_FILE, GOOD_RAW_TABLE_PREDICTION)
            self.logger.add_log(self.logs_file, "Fetched prediction data from dB.")

            self.logger.add_log(self.logs_file, "All operations of Prediction_Validation are completed !!")

        except Exception as e:
            error_message = f"Error occurred while validating  data: {str(e)}"
            self.logger.add_log(self.logs_file,error_message)
            self.logs_file.close()
            raise Exception(error_message)
        finally:
            self.logs_file.close()

