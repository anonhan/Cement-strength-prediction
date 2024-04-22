import pickle
import os
import shutil
from Prediction_Model.config.config import MODELS_DIR

class FileOperations:
    def __init__(self, logger_file, logger):
        """
        This class shall be used to save the model after training and load the saved model for prediction.

        Parameters:
        logger_file (str): File path object for logging.
        logger: Logger instance for logging.
        """
        self.log_file = logger_file
        self.logger = logger
    
    def save_model(self, model, filename):
        """
        Save a machine learning model to a file.

        Parameters:
        model: The machine learning model to save.
        filepath (str): The file path where the model will be saved.

        Returns:
        None
        """
        self.logger.add_log(self.log_file, 'Entered the save_model method of the FileOperations class.')
        try:
            if not os.path.exists(MODELS_DIR):
                os.makedirs(MODELS_DIR)
            else:
                with open(f'{MODELS_DIR}/{filename}.sav', 'wb') as f:
                    pickle.dump(model, f)
                self.logger.add_log(self.log_file, "Model saved successfully.")
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while saving the model:: {str(e)}')
            raise Exception()
    
    def load_model(self, model_name):
        """
        
        """
        self.logger.add_log(self.log_file, 'Entered the load_model method of the FileOperations class.')
        try:
            if os.path.exists(MODELS_DIR):
                with open(f'{MODELS_DIR}/{model_name}.sav','rb') as f:
                    model = pickle.load(f)
                    self.logger.add_log(self.log_file, 'Loaded model '+str(model_name))
                    return model
        except Exception as e:
            self.logger.add_log(self.log_file, "Failed to load the model "+str(model))
            self.logger.add_log(self.log_file, "Error:: "+str(e))
        
    # def find_correct_model(self, cluster_number):
    #     """
    #     Description: Selects the correct model from the cluster number.
    #     """
    #     self.logger.add_log(self.log_file, 'Entered the find_correct_model method of the FileOperations class.')
    #     try:


