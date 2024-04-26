from sklearn.model_selection import train_test_split
from Prediction_Model.data_ingestion.data_loader import Data_Loader
from Prediction_Model.data_preprocessing.data_preprocessing  import Preprocessor
from Prediction_Model.file_operations.file_methods import FileOperations
from Prediction_Model.app_logging.app_logger import App_Logger
from Prediction_Model.config.config import (PREDICTION_LOGS,
                                            PREDICTION_DATA_DIR,
                                            PREDICTION_DATA_FILE,
                                            NUMERIC_COLS_PRED,
                                            CLUSTERING_MODEL_NAME,
                                            PREDICTION_OUTPUT_DIR,
                                            PREDICTION_OUTPUT_FILE,
                                            MODELS_DIR,
                                            PREDICTION_MODELS_DIR)
import os
import pandas as pd
import traceback

class MakePredictions:
    def __init__(self):
        self.log_file = open(PREDICTION_LOGS, 'a+')
        self.logger = App_Logger()
    
    def start_prediction(self):
        self.logger.add_log(self.log_file, "Recived prediction request.")
        is_prediction_successful = False
        try:
            data = Data_Loader(file=os.path.join(PREDICTION_DATA_DIR,PREDICTION_DATA_FILE),
                               log_file_obj=self.log_file,
                               logger_obj=self.logger,
                               data_type='Prediction').get_data()

            preprocessor = Preprocessor(self.log_file, self.logger)
            is_null_present, cols_with_na = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data, numeric_cols=NUMERIC_COLS_PRED)
            # Data Transformation
            X = preprocessor.log_transform(data, NUMERIC_COLS_PRED)

            # Dividing data into clusters
            file_op = FileOperations(self.log_file, self.logger)
            k_means_model = file_op.load_model(CLUSTERING_MODEL_NAME, MODELS_DIR)
            clusters = k_means_model.predict(X)
            X['cluster'] = clusters
            list_of_clusters = X['cluster'].unique()
            result = []

            for i in list_of_clusters:
                cluster_data = X[X['cluster']==i]
                cluster_data.drop('cluster', axis=1, inplace=True)
                scaled_data = preprocessor.standardize_data(cluster_data, NUMERIC_COLS_PRED)
                model = file_op.find_correct_model(i, PREDICTION_MODELS_DIR)

                for val in cluster_data.values:
                    val_reshaped = val.reshape(1, -1)
                    result.append(model.predict(val_reshaped))

            result =  pd.DataFrame(result,columns=['Predictions'])

            if not os.path.exists(PREDICTION_OUTPUT_DIR):
                os.makedirs(PREDICTION_OUTPUT_DIR)
            result.to_csv(PREDICTION_OUTPUT_FILE)
            is_prediction_successful = True
            self.logger.add_log(self.log_file, 'Successful End of Prediction')

        except Exception as e:
            error_message = 'Error while making predictions::' + str(e)
            error_message_with_line = error_message + "\n" + traceback.format_exc()
            self.logger.add_log(self.log_file, error_message_with_line)
        finally:
            self.log_file.close()
            
        return is_prediction_successful


            