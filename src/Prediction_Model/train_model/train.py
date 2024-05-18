from sklearn.model_selection import train_test_split
from Prediction_Model.data_ingestion.data_loader import Data_Loader
from Prediction_Model.data_preprocessing.data_preprocessing import Preprocessor
from Prediction_Model.clustering.clustering import KMeansClustering
from Prediction_Model.best_model_finder.model_finder import BestModelFinder
from Prediction_Model.file_operations.file_methods import FileOperations
from Prediction_Model.app_logging.app_logger import App_Logger
from Prediction_Model.config.config import (TRAINING_LOGS, 
                                            TRAINING_DATA_DIR, 
                                            TRAINING_DATA_FILE,
                                            NUMERIC_COLS,
                                            RANDOM_SEED,
                                            PREDICTION_MODELS_DIR)
import os
import traceback

class TrainModel:
    def __init__(self):
        self.log_file = open(TRAINING_LOGS, 'a+')
        self.logger = App_Logger()

    def start_training(self):
        """
        """
        self.logger.add_log(self.log_file, "Started training the model...")
        try:
            data = Data_Loader(file=os.path.join(TRAINING_DATA_DIR,TRAINING_DATA_FILE),
                                          log_file_obj=self.log_file,
                                          logger_obj=self.logger,
                                          data_type='training').get_data()
            file_op = FileOperations(self.log_file, self.logger)
            
            # Data preprocessing
            preprocessor = Preprocessor(self.log_file, self.logger)
            is_null_present, cols_with_na = preprocessor.is_null_present(dataframe=data)
            if is_null_present:
                data = preprocessor.impute_missing_values(dataframe=data, numeric_cols=NUMERIC_COLS)
            X,y = preprocessor.separate_features_label(dataframe=data, label_column="Concrete_compressive _strength")
            X = preprocessor.log_transform(dataframe=X, numeric_cols=NUMERIC_COLS[:-1])

            # Dividing data into clusters
            kmeans = KMeansClustering(self.log_file, self.logger)
            number_of_clusters = kmeans.elbow_plot(data=X)
            X = kmeans.create_clusters(dataframe=X, optimal_clusters=number_of_clusters)
            X['label'] = y
            # List of clusters
            list_of_clusters = X['cluster'].unique()

            for i in list_of_clusters:
                self.logger.add_log(self.log_file, f"Training on the cluster number {i}")
                cluster_data = X[X['cluster']==i]
                cluster_features = cluster_data.drop(['label','cluster'],axis=1)
                cluster_label = cluster_data['label']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=RANDOM_SEED)
                x_train_scaled = preprocessor.standardize_data(x_train, NUMERIC_COLS[:-1])
                x_test_scaled = preprocessor.standardize_data(x_test, NUMERIC_COLS[:-1])

                model_finder = BestModelFinder(log_file=self.log_file,
                                               logger=self.logger,
                                               X_train=x_train_scaled,
                                               y_train=y_train,
                                               X_test=x_test_scaled,
                                               y_test=y_test,
                                               cluster_number=i
                                               )
                best_model_name, best_model = model_finder.find_best_model()
                best_model_name, best_model = model_finder.optimize_best_model(best_model_name, best_model)


                # Saving the best model
                file_op.save_model(best_model, best_model_name+str(i), PREDICTION_MODELS_DIR)
            
            # logging the successful Training
            self.logger.add_log(self.log_file, 'Successful End of Training')
            self.log_file.close()
        
        except Exception as e:
            error_message = 'Error while making predictions::' + str(e)
            error_message_with_line = error_message + "\n" + traceback.format_exc()
            self.logger.add_log(self.log_file, error_message_with_line)
            raise e