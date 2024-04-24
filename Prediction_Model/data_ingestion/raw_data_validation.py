import json
import os
import shutil
import re
import pandas as pd
from Prediction_Model.config.config import PACAKAGE_ROOT
from Prediction_Model.app_logging.app_logger import App_Logger

class Raw_Data_Validation:
    def __init__(self, raw_files_path, json_schema_file):
        self.raw_files_path = raw_files_path
        self.logger = App_Logger()
        # self.json_schema = "schema_training.json"
        self.json_schema = os.path.join(PACAKAGE_ROOT, json_schema_file)
    
    def get_values_from_schema(self, data_ingestion_logs_path):
        """
        Description: Function to get the values from the JSON Schema to validate training data files.
        """
        try:
            with open(self.json_schema, 'r') as f:
                dict_schema = json.load(f)
                f.close()
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "JSON Schema loaded.")
            
            LengthOfDateStampInFile = dict_schema['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dict_schema['LengthOfTimeStampInFile']
            NumberofColumns = dict_schema['NumberofColumns']
            ColName = dict_schema['ColName']
            logs_file.close()

        except Exception as e:
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Error while reading JSON Schema :: "+str(e))
            logs_file.close()
            raise Exception()
        
        return LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName
        
    def create_good_bad_archive_data_dir(self, data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir):
        """
        Description: Method to create the Good, Bad and Archive Data folder if does not exists.
        """
        try:
            if not os.path.exists(good_raw_dir):
                os.makedirs(good_raw_dir)
            if not os.path.exists(bad_raw_dir):
                os.makedirs(bad_raw_dir)
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Created Good, Bad and Archive Data folders.")
            logs_file.close()
        except Exception as e:
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Error occurred while creating Good, Bad and Archive Data folder :: "+str(e))
            logs_file.close()
            raise Exception()
        
    def remove_good_bad_dirs(self, data_ingestion_logs_path, good_raw_dir, bad_raw_dir):
        """
        Description: This method deletes the Good and Bad raw directories once the data is inserted in the DB.
        Created one for both can be created separately.
        """
        try:
            if os.path.exists(good_raw_dir):
                shutil.rmtree(good_raw_dir)
            if os.path.exists(bad_raw_dir):
                shutil.rmtree(bad_raw_dir)
            logs_file = open(data_ingestion_logs_path, 'a+')
            self.logger.add_log(logs_file, "Removed Good and Bad data folders.")
            logs_file.close()
        except Exception as e:
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Error occurred while deleting Good and Bad data folder :: "+str(e))
            logs_file.close()
            raise Exception()

    def move_bad_files_to_archive(self, data_ingestion_logs_path, bad_raw_dir, archive_dir):
        """
        Description: Method to move the bad raw files to archive.
        """
        try:
            if os.path.exists(bad_raw_dir):
                # Iterate over all the files in Bad Dir and move to Archive
                for filename in os.listdir(bad_raw_dir):
                    csv_file = os.path.join(bad_raw_dir, filename)
                    if os.path.isfile(csv_file) and filename.lower().endswith('.csv'):
                        shutil.move(csv_file, archive_dir)                
        except Exception as e:
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Error occurred while moving bad files to archive folder :: "+str(e))
            logs_file.close()
            raise Exception()
    
    def validate_file_name(self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile, data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir, files_dir):
        """
        Description: Function to validate the name of the file, if matches then the data file
                     will be kept in Good raw folder else copied to the Bad raw folder.
        """
        def match_pattern(file_name):
            pattern = regex % (LengthOfDateStampInFile, LengthOfTimeStampInFile)
            return bool(re.match(pattern, file_name))

        self.remove_good_bad_dirs(data_ingestion_logs_path, good_raw_dir, bad_raw_dir)
        self.create_good_bad_archive_data_dir(data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir)
        logs_file = open(data_ingestion_logs_path, 'a+')
        try:
            files = [file for file in os.listdir(files_dir)]
            for filename in files:
                if match_pattern(filename):
                    shutil.copy(f"{files_dir}/{filename}", good_raw_dir)
                    self.logger.add_log(logs_file, "Valid file name! File moved to Good Raw folder::"+str(filename))
                else:
                    shutil.copy(f"{files_dir}/{filename}", bad_raw_dir)
                    self.logger.add_log(logs_file, "Invalid file name! File moved to Bad Raw folder::"+str(filename))
            logs_file.close()

        except Exception as e:
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Error occurred while validating training files name::"+str(e))
            logs_file.close()
            raise Exception()

    def validate_column_names(self, NumberofColumns, ColName, data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir):
        """
        Description: Method to read the CSV files to validate the name of the columns and the number of columns
                     from Good Raw folder. If the names of the columns is invalid the fole will be moved to the
                     Bad Raw Folder.
        """
        schema_col_names = ColName.keys()
        logs_file = open(f"{data_ingestion_logs_path}", 'a+')
        try:
            if os.path.exists(good_raw_dir):
                files = [file for file in os.listdir(good_raw_dir)]
                for filename in files:
                    csv_file = pd.read_csv(good_raw_dir+"/"+filename)
                    column_names = csv_file.columns

                    if csv_file.shape[1] == NumberofColumns:
                        if set(schema_col_names).issubset(column_names):
                            pass
                        else:
                            shutil.move(good_raw_dir+"/"+filename, bad_raw_dir)
                            self.logger.add_log(logs_file, "Invalid column headings! File moved to Bad Raw folder::"+str(filename))   
                    else:
                        shutil.move(good_raw_dir+"/"+filename, bad_raw_dir)
                        self.logger.add_log(logs_file, "Invalid number of columns! File moved to Bad Raw folder::"+str(filename))                        


        except Exception as e:
            logs_file = open(f"{data_ingestion_logs_path}", 'a+')
            self.logger.add_log(logs_file, "Error occurred while validating training file columns::"+str(e))
            logs_file.close()
            raise Exception()


    # def validate_missing_values_column(self):
    #     """
    #     Description: Method to validate the missing valuesi n
    #     """


# rw = Raw_Data_Validation('abc')
# print(rw.create_good_bad_archive_data_dir(TRAINING_DATA_INGESTION_LOGS_FILE, GOOD_RAW_DIR_TRAIN, BAD_RAW_DIR_PRED, ARCHIVE_DIR_TRAIN))
# # print(rw.move_bad_files_to_archive())
# # print(rw.validate_file_name(FILE_NAME_PATTERN, 8,6))
# # d = {
# # 		"Cement _component_1" : "FLOAT",
# # 		"Blast Furnace Slag _component_2" : "FLOAT",
# # 		"Fly Ash _component_3" : "FLOAT",
# # 		"Water_component_4" : "FLOAT",
# # 		"Superplasticizer_component_5" : "FLOAT",
# # 		"Coarse Aggregate_component_6" : "FLOAT",
# # 		"Fine Aggregate_component_7" : "FLOAT",
# # 		"Age_day" : "INTEGER",
# # 		"Concrete_compressive _strength" : "FLOAT"}
# # print(rw.validate_column_names(9, d))