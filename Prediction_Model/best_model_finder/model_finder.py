import optuna
import mlflow
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from Prediction_Model.config.config import MLFLOW_URI

class BestModelFinder:
    def __init__(self, log_file, logger, X_train, y_train, X_test, y_test):
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

    def objective(self, trial, cluster_number):
        """
        Objective function for hyperparameter tuning.

        Parameters:
        trial: A single optimization trial.

        Returns:
        float: R2 score of the model.
        """
        try:
            mlflow.set_tracking_uri(uri=MLFLOW_URI)
            mlflow.set_experiment(f"Cement-strenght-prediction-{cluster_number}")
            with mlflow.start_run():
                model_name = trial.suggest_categorical('model', ['linear', 'random_forest', 'xgboost'])
                mlflow.log_params(trial.params)

                if model_name == 'linear':
                    model = LinearRegression()

                elif model_name == 'random_forest':
                    n_estimators = trial.suggest_int('n_estimators_rf', 10, 100)
                    max_depth = trial.suggest_int('max_depth_rf', 2, 32)
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

                else:
                    param = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'n_estimators': trial.suggest_int('n_estimators_xgb', 10, 100),
                        'max_depth': trial.suggest_int('max_depth_xgb', 2, 32),
                        'learning_rate': trial.suggest_loguniform('learning_rate_xgb', 0.001, 0.1),
                    }
                    model = xgb.XGBRegressor(**param, random_state=42)

                # Perform cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                for train_index, test_index in kf.split(self.X_train):
                    x_tr, x_ts = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                    y_tr, y_ts = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

                    model.fit(x_tr, y_tr)
                    y_pred = model.predict(x_ts)
                    score = r2_score(y_ts, y_pred)
                    scores.append(score)

                avg_score = sum(scores) / len(scores)
                mlflow.log_metric('r2_score', avg_score)
                mlflow.log_params(model.get_params())

                return avg_score

        except Exception as e:
            error_message = f'Error occurred while creating the objective method: {str(e)}'
            self.logger.add_log(self.log_file, error_message)
            raise Exception(error_message)
    
    def optimize(self, cluster_number):
        """
        Perform hyperparameter optimization.

        Returns:
        tuple: Best model name and best model instance.
        """
        try:
            self.logger.add_log(self.log_file, 'Starting optimization process.')
            study = optuna.create_study(direction='maximize')
            partial_objective = partial(self.objective,cluster_number=cluster_number)
            study.optimize(partial_objective, n_trials=50)
            best_params = study.best_params
            best_model = None
            mlflow.set_experiment('Cement-strength-best-models')
            with mlflow.start_run():
                if best_params['model'] == 'linear':
                    best_model = LinearRegression()

                elif best_params['model'] == 'random_forest':
                    best_model = RandomForestRegressor(n_estimators=best_params['n_estimators_rf'], max_depth=best_params['max_depth_rf'])

                else:
                    best_model = xgb.XGBRegressor(n_estimators=best_params['n_estimators_xgb'], max_depth=best_params['max_depth_xgb'], learning_rate=best_params['learning_rate_xgb'])

                self.logger.add_log(self.log_file, 'Optimization process completed successfully.')
                self.logger.add_log(self.log_file, f'Best model is {best_params["model"]}')
                
                # Training and prediction on best model
                best_model.fit(self.X_train, self.y_train)
                y_pred = best_model.predict(self.X_test)
                score = r2_score(self.y_test, y_pred)
                mlflow.log_metric('r2_score', score)
                mlflow.log_params(best_model.get_params())

            return best_params['model'], best_model

        except Exception as e:
            error_message = f'Error occurred during model optimization: {str(e)}'
            self.logger.add_log(self.log_file,error_message)
            raise Exception(error_message)
