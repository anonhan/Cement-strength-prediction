import json
import os
import shutil
import re
import pandas as pd
from Prediction_Model.config.config import PACKAGE_ROOT
from Prediction_Model.app_logging.app_logger import App_Logger
import traceback

class Raw_Data_Validation:
    def __init__(self, raw_files_path, json_schema_file):
        """
        Initializes Raw_Data_Validation object.

        Parameters:
        raw_files_path (str): Path to the directory containing raw data files.
        json_schema_file (str): Name of the JSON schema file.

        """
        self.raw_files_path = raw_files_path
        self.logger = App_Logger()
        # self.json_schema = "schema_training.json"
        self.json_schema = os.path.join(PACKAGE_ROOT, json_schema_file)
    
    def get_values_from_schema(self, data_ingestion_logs_path):
        """
        Retrieves values from the JSON Schema to validate training data files.

        Parameters:
        data_ingestion_logs_path (str): Path to the data ingestion logs file.

        Returns:
        tuple: A tuple containing LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName.

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
        Creates the Good, Bad and Archive Data folders if they do not exist.

        Parameters:
        data_ingestion_logs_path (str): Path to the data ingestion logs file.
        good_raw_dir (str): Path to the directory to store good raw data files.
        bad_raw_dir (str): Path to the directory to store bad raw data files.
        archive_dir (str): Path to the directory to store archived data files.

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
        Deletes the Good and Bad raw directories once the data is inserted in the DB.

        Parameters:
        data_ingestion_logs_path (str): Path to the data ingestion logs file.
        good_raw_dir (str): Path to the directory storing good raw data files.
        bad_raw_dir (str): Path to the directory storing bad raw data files.

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
        Moves the bad raw files to archive.

        Parameters:
        data_ingestion_logs_path (str): Path to the data ingestion logs file.
        bad_raw_dir (str): Path to the directory storing bad raw data files.
        archive_dir (str): Path to the directory to store archived data files.

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
        Validates the name of the file and moves it to the Good Raw folder if it matches the pattern,
        otherwise copies it to the Bad Raw folder.

        Parameters:
        regex (str): Regular expression pattern for file name validation.
        LengthOfDateStampInFile (int): Length of date stamp in file name.
        LengthOfTimeStampInFile (int): Length of timestamp in file name.
        data_ingestion_logs_path (str): Path to the data ingestion logs file.
        good_raw_dir (str): Path to the directory to store good raw data files.
        bad_raw_dir (str): Path to the directory to store bad raw data files.
        archive_dir (str): Path to the directory to store archived data files.
        files_dir (str): Path to the directory containing the files to be validated.

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
            self.logger.add_log(logs_file, "Error occurred while validating file name::"+str(e))
            logs_file.close()
            raise Exception()
        
        
    def validate_file_name_without_path(self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile, data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir, uploaded_file, file_name):
        """
        Validates the name of the file and moves it to the Good Raw folder if it matches the pattern,
        otherwise copies it to the Bad Raw folder.

        Parameters:
        regex (str): Regular expression pattern for file name validation.
        LengthOfDateStampInFile (int): Length of date stamp in file name.
        LengthOfTimeStampInFile (int): Length of timestamp in file name.
        data_ingestion_logs_path (str): Path to the data ingestion logs file.
        good_raw_dir (str): Path to the directory to store good raw data files.
        bad_raw_dir (str): Path to the directory to store bad raw data files.
        archive_dir (str): Path to the directory to store archived data files.
        uploaded_file (file): File object containing the uploaded file data.
        file_name (str): Name of the uploaded file.

        """
        def match_pattern(file_name):
            pattern = regex % (LengthOfDateStampInFile, LengthOfTimeStampInFile)
            return bool(re.match(pattern, file_name))

        self.remove_good_bad_dirs(data_ingestion_logs_path, good_raw_dir, bad_raw_dir)
        self.create_good_bad_archive_data_dir(data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir)
        logs_file = open(data_ingestion_logs_path, 'a+')
        try:
            # Read the BytesIO object as pandas DataFrame
            df = pd.read_csv(uploaded_file)

            # Get the file name from the uploaded file
            filename = file_name
            if match_pattern(filename):
                # Copy the file to the good_raw_dir
                with open(f"{good_raw_dir}/{filename}", "wb") as good_file:
                    good_file.write(uploaded_file.getvalue())
                self.logger.add_log(logs_file, "Valid file name! File moved to Good Raw folder::"+str(filename))
            else:
                # Copy the file to the bad_raw_dir
                with open(f"{bad_raw_dir}/{filename}", "wb") as bad_file:
                    bad_file.write(uploaded_file.getvalue())
                self.logger.add_log(logs_file, "Invalid file name! File moved to Bad Raw folder::"+str(filename))

            logs_file.close()

        except Exception as e:
            error_message = traceback.format_exc()
            self.logger.add_log(logs_file, "Error occurred: " + error_message)
            raise Exception()

    def validate_column_names(self, NumberofColumns, ColName, data_ingestion_logs_path, good_raw_dir, bad_raw_dir, archive_dir):
        """
        Validates the column names of CSV files and moves the invalid files to the Bad Raw folder.

        Parameters:
        NumberofColumns (int): Expected number of columns in the CSV files.
        ColName (dict): Dictionary containing column names.
        data_ingestion_logs_path (str): Path to the data ingestion logs file.
        good_raw_dir (str): Path to the directory storing good raw data files.
        bad_raw_dir (str): Path to the directory storing bad raw data files.
        archive_dir (str): Path to the directory to store archived data files.

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
