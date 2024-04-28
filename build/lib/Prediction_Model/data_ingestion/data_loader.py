import pandas as pd

class Data_Loader:
    def __init__(self, file, log_file_obj, logger_obj, data_type):
        """
        Initialize Data_Loader object.

        Parameters:
        file (str): File path of the data source.
        log_file_obj: Log file object.
        logger_obj: Logger object.
        data_type (str): Type of data being loaded (e.g., 'training', 'testing', etc.).
        """
        self.file = file
        self.log_file_obj = log_file_obj
        self.logger_obj = logger_obj
        self.data_type = data_type

    def get_data(self):
        """
        Read data from the source file.

        Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
        """
        try:
            self.logger_obj.add_log(self.log_file_obj, f"Entered into get_data function of Data_Loader class for {self.data_type} data.")
            self.data = pd.read_csv(self.file)
            self.logger_obj.add_log(self.log_file_obj, f"{self.data_type.capitalize()} data read successfully.")
            return self.data
        except Exception as e:
            error_message = f"Error occurred while reading {self.data_type} data: {str(e)}"
            self.logger_obj.add_log(self.log_file_obj, error_message)
            raise Exception(error_message)
