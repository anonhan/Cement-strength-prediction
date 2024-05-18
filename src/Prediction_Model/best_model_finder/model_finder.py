import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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
    
    def optimize(self, cluster_number):
        """
        Perform hyperparameter optimization using GridSearchCV.

        Returns:
        tuple: Best model name and best model instance.
        """
        try:
            self.logger.add_log(self.log_file, 'Starting optimization process.')

            # Define models and parameter grids
            models = {
                'RandomForest': (RandomForestRegressor(),
                                {'n_estimators': [50, 100, 150],
                                'max_depth': [None, 10, 20],
                                'min_samples_split': [2, 5, 10],
                                'n_jobs': [2]}),
                'SVR': (SVR(),
                        {'kernel': ['linear', 'rbf'],
                        'C': [1, 10, 100],
                        'epsilon': [0.1, 0.01, 0.001]
                        }),
                'LinearRegression': (LinearRegression(),
                                    {}),
                'GradientBoostingRegressor': (GradientBoostingRegressor(),
                                            {'n_estimators': [50, 100],
                                                'learning_rate': [0.05, 0.01, 0.001],
                                                'max_depth': [3, 5, 10]
                                                })
            }

            best_model_name = None
            best_score = float('-inf')
            best_model_instance = None
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            for model_name, (model, param_grid) in models.items():
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(self.X_train, self.y_train)

                # Log best model if it has better score
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model_name = model_name
                    best_model_instance = grid_search.best_estimator_

                    self.logger.add_log(self.log_file, 'Optimization process completed successfully.')
                    self.logger.add_log(self.log_file, f'Best model is {model_name}')

            # Log parameters and metrics for the best model
            with mlflow.start_run():
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({'neg_mean_squared_error': best_score})

                # Log the best model
                custom_model_name = "prediction_model"+"_"+str(self.cluster_number)
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