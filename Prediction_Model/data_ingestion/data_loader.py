import pandas as pd

class Data_Loader:
    def __init__(self, file, log_file_obj, logger_obj, data_type):
        self.file = file
        self.log_file_obj = log_file_obj
        self.logger_obj = logger_obj
        self.data_type = data_type

    def get_data(self):
        """
        Description: This function reads the data from source.
        """
        self.logger_obj.add_log(self.log_file_obj, "Entered into get_data function of Data_Loader class.")
        try:
            self.data = pd.read_csv(self.file,)
            self.logger_obj.add_log(self.log_file_obj, f"{self.data_type} data read successfull.")
            return self.data
        except Exception as e:
            self.logger_obj.add_log(self.log_file_obj, f"Error while reading {self.data_type} data::"+str(e))
            raise Exception()
