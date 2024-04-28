import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, logger_file, logger):
        """
        Initialize Preprocessor object with logger file and logger instance.

        Parameters:
        logger_file (str): File path object for logging.
        logger: Logger instance for logging.
        """
        self.log_file = logger_file
        self.logger = logger

    def drop_columns(self, dataframe, columns):
        """
        Drop specified columns from the dataframe.

        Parameters:
        dataframe (pandas.DataFrame): Input DataFrame.
        columns (list): List of column names to be dropped.

        Returns:
        pandas.DataFrame: DataFrame after dropping specified columns.
        """
        self.logger.add_log(self.log_file, 'Entered the drop_columns method of the Preprocessor class.')
        self.data = dataframe.copy()
        try:
            required_data = self.data.drop(labels=columns, axis=1)
            self.logger.add_log(self.log_file, f'Dropped columns: {columns} from the dataframe.')
            return required_data

        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while dropping column from dataframe:: {str(e)}')
            raise Exception()

    def separate_features_label(self, dataframe, label_column):
        """
        Separate data into Features and Labels.

        Parameters:
        dataframe (pandas.DataFrame): Input DataFrame.
        label_column (str): Column name to be treated as label.

        Returns:
        tuple: X (Features DataFrame), y (Labels Series).
        """
        self.logger.add_log(self.log_file, 'Entered into the separate_features_label function of Preprocessor class.')
        try:
            self.data = dataframe.copy()
            self.X = self.data.drop(labels=label_column, axis=1)
            self.y = self.data[label_column]
            self.logger.add_log(self.log_file, f'Separated label column: {label_column} from the dataframe into X and y.')
            return self.X, self.y

        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while separating the features and labels:: {str(e)}')
            raise Exception()

    def replace_invalid_values_with_null(self, dataframe):
        """
        Replace invalid values with null values.

        Parameters:
        dataframe (pandas.DataFrame): Input DataFrame.

        Returns:
        pandas.DataFrame: DataFrame after replacing invalid values with nulls.
        """
        self.logger.add_log(self.log_file, 'Entered into the replace_invalid_values_with_null function of Preprocessor class.')
        self.data = dataframe.copy()
        try:
            for column in self.data.columns:
                count = self.data[column][self.data[column] == '?'].count()
                if count != 0:
                    self.data[column] = self.data[column].replace('?', np.nan)
                    self.logger.add_log(self.log_file, 'Replaced "?" with NaNs.')
                return self.data
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while replacing invalid values:: {str(e)}')
            raise Exception()

    def is_null_present(self, dataframe):
        """
        Check if null values are present in the DataFrame.

        Parameters:
        dataframe (pandas.DataFrame): Input DataFrame.

        Returns:
        tuple: (is_null (bool), null_columns (list))
               is_null: True if null values are present, False otherwise.
               null_columns: List of columns with null values.
        """
        self.logger.add_log(self.log_file, 'Entered into the is_null_present function of Preprocessor class.')
        data = dataframe.copy()
        is_null = False
        try:
            null_columns = data.columns[data.isnull().any()].tolist()
            if not null_columns:
                return is_null, null_columns
            else:
                return is_null, null_columns
            
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while checking nulls in columns :: {str(e)}')
            raise Exception()

    def standardize_data(self, dataframe, numeric_cols):
        """
        Standardize numeric variables in a DataFrame.

        Parameters:
        dataframe (pandas.DataFrame): DataFrame containing numeric variables.
        numeric_cols (list): List of column names containing numeric variables.

        Returns:
        pandas.DataFrame: DataFrame with standardized numeric variables.
        """
        try:
            self.logger.add_log(self.log_file, 'Entered into the standardize_data function of Preprocessor class.')
            scaler = StandardScaler()

            # Fit and transform the scaler on the DataFrame
            df_standardized = dataframe.copy()
            df_standardized[numeric_cols] = scaler.fit_transform(df_standardized[numeric_cols])
            self.logger.add_log(self.log_file, "Standardized the data.")
            return df_standardized
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while standardizing data :: {str(e)}')
            raise Exception()

    def log_transform(self, dataframe, numeric_cols):
        """
        Apply log transformation to numeric variables in a DataFrame.

        Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing numeric variables to transform.
        numeric_cols (list): List of column names containing numeric variables.

        Returns:
        pandas.DataFrame: DataFrame with numeric variables log-transformed.
        """
        try:
            self.logger.add_log(self.log_file, 'Entered into the log_transform function of Preprocessor class.')
            df_transformed = dataframe.copy()

            for col in numeric_cols:
                df_transformed[col] = np.log(df_transformed[col] + 1)
            self.logger.add_log(self.log_file, "Log transformed the data.")
            return df_transformed
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while log transformation :: {str(e)}')
            raise Exception()

    def impute_missing_values(self, dataframe, numeric_cols, n_neighbors=5):
        """
        Impute missing values in numeric variables of a DataFrame using k-nearest neighbors (KNN) algorithm.

        Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing numeric variables with missing values.
        numeric_cols (list): List of column names containing numeric variables with missing values.
        n_neighbors (int, optional): Number of neighbors to use for imputation. Defaults to 5.

        Returns:
        pandas.DataFrame: DataFrame with missing values imputed using KNN.
        """
        self.logger.add_log(self.log_file, 'Entered into the impute_missing_values function of Preprocessor class.')
        imputer = KNNImputer(n_neighbors=n_neighbors)
        try: 
            # Impute missing values in numeric columns
            df_imputed = dataframe.copy()
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
            self.logger.add_log(self.log_file, "Imputed the missing valued in the data.")
            return df_imputed
        except Exception as e:
            self.logger.add_log(self.log_file, f'Error while imputing missing values :: {str(e)}')
            raise Exception()
        