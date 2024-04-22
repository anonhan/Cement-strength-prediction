import os
import csv
import mysql
import mysql.connector
import configparser
from Prediction_Model.app_logging.app_logger import App_Logger
from Prediction_Model.config.config import (PACAKAGE_ROOT, 
                                            GOOD_RAW_DIR, 
                                            BAD_RAW_DIR, 
                                            DATA_INGESTION_LOGS_FILE, 
                                            GOOD_RAW_TABLE,
                                            TRAINING_DATA_DIR,
                                            TRAINING_DATA_FILE,
                                            CHUNK_SIZE,
                                            DATA_DIR
                                            )

# Create the conig object
config = configparser.ConfigParser()
config.read(f"{PACAKAGE_ROOT}/user_password.ini")

# Access configuration data
host = config['Database']['Host']
username = config['Database']['Username']
password = config['Database']['Password']
database = config['Database']['Database']

class dBOperations:
    def __init__(self):
        self.logger = App_Logger()
        
    def connect_to_db(self):
        """
        Description: Establish a connection to the MySQL database.
        """
        try:
            connection = mysql.connector.connect(
                host = host,
                database = database,
                user = username,
                password = password
            )
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,"Connected to MySQL database successfully.")
            logs_file.close()
            return connection
        except mysql.connector.Error as e:
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,"Error while connecting to MySQL database::"+str(e))
            logs_file.close()
            return None
    
    def create_table(self, column_names):
        """
        Description: Method to create the table in the db.
        """
        try:
            table_name = GOOD_RAW_TABLE
            conn = self.connect_to_db()
            cursor = conn.cursor()
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            column_definitions = ", ".join([f"`{name}` {data_type}" for name, data_type in column_names.items()])
            # Construct the CREATE TABLE statement
            create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({column_definitions})"
            # Execute the CREATE TABLE statement
            cursor.execute(create_table_query)
            conn.commit()
            # create log
            self.logger.add_log(logs_file ,"Created Table successfully.")
            logs_file.close()

        except Exception as e:
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,"Error while table creation in dB::"+str(e))
            logs_file.close()
            conn.rollback()

        # Close the cursor and connection
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def insert_good_data_into_db(self):
        """
        Description: Insert the good raw data into Db by reading the files from good data folder.
        """   
        try:
            files = [file for file in os.listdir(GOOD_RAW_DIR)]
            conn = self.connect_to_db()
            cursor = conn.cursor()
            table_name = GOOD_RAW_TABLE
            
            for filename in files:
                with open(f"{GOOD_RAW_DIR}/{filename}", "r") as csv_file:
                    csv_reader = csv.reader(csv_file)
                    next(csv_reader)
                    for row in csv_reader:
                        insert_query = f"INSERT INTO {table_name} VALUES ({','.join(['%s'] * len(row))})"
                        cursor.execute(insert_query, row)
            conn.commit()
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,"Inserted good raw data into dB successfully.")

        except Exception as e:
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,"Error while inserting data into dB table::"+str(e))
            logs_file.close()
            conn.rollback()
            raise Exception()
        
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
        
    def select_data_from_table(self):
        """
        Description: Method to select the data from the dB tables and returns as CSV
        """
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            chunk_size = CHUNK_SIZE
            if not os.path.exists(TRAINING_DATA_DIR):
                os.makedirs(TRAINING_DATA_DIR)
            with open(f"{TRAINING_DATA_DIR}/{TRAINING_DATA_FILE}", "w", newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Get table columns
                cursor.execute(f"SHOW COLUMNS FROM {GOOD_RAW_TABLE};")
                headers = [column[0] for column in cursor.fetchall()]
                csv_writer.writerow(headers)

                # Fetch data in chunks and write to CSV
                offset = 0
                while True:
                    cursor.execute(f"SELECT * FROM {GOOD_RAW_TABLE} LIMIT {chunk_size} OFFSET {offset}")
                    rows = cursor.fetchall()
                    if not rows:
                        break
                    csv_writer.writerows(rows)
                    offset += chunk_size

            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,f"Fetched data from {GOOD_RAW_TABLE} table.")
            logs_file.close()

        except Exception as e:
            logs_file = open(f"{DATA_INGESTION_LOGS_FILE}", 'a+')
            self.logger.add_log(logs_file ,"Error while selecting data from dB::"+str(e))
            logs_file.close()
            raise Exception()

        finally:
            if conn.is_connected():
                conn.close()
                cursor.close()



# db = dBOperations()
# d = {
# 		"Cement _component_1" : "FLOAT",
# 		"Blast Furnace Slag _component_2" : "FLOAT",
# 		"Fly Ash _component_3" : "FLOAT",
# 		"Water_component_4" : "FLOAT",
# 		"Superplasticizer_component_5" : "FLOAT",
# 		"Coarse Aggregate_component_6" : "FLOAT",
# 		"Fine Aggregate_component_7" : "FLOAT",
# 		"Age_day" : "INTEGER",
# 		"Concrete_compressive _strength" : "FLOAT"}
# # print(db.connect_to_db())
# # print(db.create_table(d))
# # print(db.insert_good_data_into_db())
# print(db.select_data_from_table())