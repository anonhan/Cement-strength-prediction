import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from Prediction_Model.config.config import MLFLOW_URI, MLFLOW_EXPERIMENT_NAME

class BestModelFinder:
    def __init__(self, log_file, logger, X_train, y_train, X_test, y_test, cluster_number):
        """
        Initialize the BestModelFinder object.

        Parameters:
        log_file (str): File path for logging.
        logger: Logger instance for logging.
        X_train (array-like): Training input samples.
        y_train (array-like): Target values for training.
        X_test (array-like): Test input samples.
        y_test (array-like): Target values for testing.
        """
        self.log_file = log_file
        self.logger = logger
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cluster_number = cluster_number
        mlflow.set_tracking_uri(uri=MLFLOW_URI)
    
    def find_best_model(self):
        """
        Train models without hyperparameter tuning and find the best model.

        Returns:
        tuple: Best model name and best model instance.
        """
        try:
            self.logger.add_log(self.log_file, 'Starting model comparison process.')

            # Define models
            models = {
                'RandomForest': RandomForestRegressor(),
                'SVR': SVR(),
                'LinearRegression': LinearRegression(),
                'GradientBoostingRegressor': GradientBoostingRegressor()
            }

            best_model_name = None
            best_score = float('-inf')
            best_model_instance = None

            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            # Train each model and evaluate on the validation set
            for model_name, model in models.items():
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                score = r2_score(self.y_test, y_pred)

                self.logger.add_log(self.log_file, f'{model_name} R2 score: {score}')

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model_instance = model

            self.logger.add_log(self.log_file, f'Best model before tuning: {best_model_name} with R2 score: {best_score}')

            return best_model_name, best_model_instance

        except Exception as e:
            error_message = f'Error occurred during model comparison: {str(e)}'
            self.logger.add_log(self.log_file, error_message)
            raise Exception(error_message)

    def optimize_best_model(self, best_model_name, best_model_instance):
        """
        Perform hyperparameter optimization using GridSearchCV on the best model.

        Returns:
        tuple: Best model name and best model instance after hyperparameter tuning.
        """
        try:
            self.logger.add_log(self.log_file, f'Starting hyperparameter tuning for the best model: {best_model_name}.')

            # Define parameter grids for hyperparameter tuning
            param_grids = {
                'RandomForest': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'n_jobs': [2]
                },
                'SVR': {
                    'kernel': ['linear', 'rbf'],
                    'C': [1, 10, 100],
                    'epsilon': [0.1, 0.01, 0.001]
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.01, 0.001],
                    'max_depth': [3, 5, 10]
                }
            }

            if best_model_name in param_grids:
                grid_search = GridSearchCV(best_model_instance, param_grids[best_model_name], cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(self.X_train, self.y_train)

                best_model_instance = grid_search.best_estimator_
                best_score = grid_search.best_score_

                self.logger.add_log(self.log_file, f'Hyperparameter tuning completed for {best_model_name}.')
                self.logger.add_log(self.log_file, f'Best params: {grid_search.best_params_}')
                self.logger.add_log(self.log_file, f'Best score: {best_score}')

                # Log parameters and metrics for the best model
                with mlflow.start_run():
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metrics({'neg_mean_squared_error': best_score})

                    # Log the best model
                    custom_model_name = "prediction_model" + "_" + str(self.cluster_number)
                    mlflow.sklearn.log_model(sk_model=best_model_instance, artifact_path=custom_model_name)
                    mlflow.set_tag("model_name", custom_model_name)

                    # Evaluate on test data
                    y_pred = best_model_instance.predict(self.X_test)
                    r2 = r2_score(self.y_test, y_pred)
                    mlflow.log_metrics({'r2_score': r2})

            return best_model_name, best_model_instance

        except Exception as e:
            error_message = f'Error occurred during model optimization: {str(e)}'
            self.logger.add_log(self.log_file, error_message)
            raise Exception(error_message)
