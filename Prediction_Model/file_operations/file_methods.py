import pickle
import os
import shutil

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
        self.is_model_dir_empty = False
    
    def save_model(self, model, filename, model_filepath):
        """
        Save a machine learning model to a file.

        Parameters:
        model: The machine learning model to save.
        model_filepath (str): The file path where the model will be saved.

        Returns:
        None
        """
        self.logger.add_log(self.log_file, 'Entered the save_model method of the FileOperations class.')
        try:
            if not os.path.exists(model_filepath):
                os.makedirs(model_filepath)
                self.is_model_dir_empty = True 
            elif not self.is_model_dir_empty:
                # Remove existing files within the directory
                for file_name in os.listdir(model_filepath):
                    os.remove(os.path.join(model_filepath, file_name))
                self.is_model_dir_empty = True 
            # Save the model
            with open(f'{model_filepath}/{filename}.sav', 'wb') as f:
                pickle.dump(model, f)
            self.logger.add_log(self.log_file, "Model saved successfully.")
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while saving the model:: {str(e)}')
            raise Exception()
    
    def load_model(self, model_name, model_filepath):
        """
        
        """
        self.logger.add_log(self.log_file, 'Entered the load_model method of the FileOperations class.')
        try:
            if os.path.exists(model_filepath):
                with open(f'{model_filepath}/{model_name}.sav','rb') as f:
                    model = pickle.load(f)
                    self.logger.add_log(self.log_file, 'Loaded model '+str(model_name))
                    return model
        except Exception as e:
            self.logger.add_log(self.log_file, "Failed to load the model "+str(model_name))
            self.logger.add_log(self.log_file, "Error:: "+str(e))
            raise Exception()
        
    def find_correct_model(self, cluster_number, model_filepath):
        """
        Description: Selects the correct model from the cluster number.
        """
        self.logger.add_log(self.log_file, 'Entered the find_correct_model method of the FileOperations class.')
        model = None
        try:
            all_models = os.listdir(model_filepath)
            for i in all_models:
                splited_name = i.split('.')[0]
                if splited_name.endswith(str(cluster_number)):  
                    model = self.load_model(splited_name, model_filepath)
                    break
            if model is None:
                self.logger.add_log(self.log_file, f"No model found for cluster number {cluster_number}.")
            return model

        except Exception as e:
            self.logger.add_log(self.log_file, f"Error: Failed to load the correct model: {e}")
            raise  Exception()


