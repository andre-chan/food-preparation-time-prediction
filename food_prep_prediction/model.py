"""
Model class for training, evaluation and prediction with Random Forest regressor, given numerical/categorical features.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True


class RandomForestModel:
    def __init__(self):
        super().__init__()
        self._log = logging.getLogger()
        self.model = None
        self.feature_names = None

    def train(self, features: pd.DataFrame, labels: pd.Series, tune_hyperparameters: bool = True, cv_iterations=100,
              cv_folds=5, **kwargs):
        """
        Fits RandomForestRegressor to X=features, y=labels.

        tune_hyperparameters: (bool)
            If True, tunes hyperparameters using Random Search CV on the grid created by self._create_params_grid()
            If False, fits RandomForestRegressor using hyperparameters provided as **kwargs.
        """
        params_grid = self._create_params_grid()
        self.feature_names = features.columns
        if tune_hyperparameters:
            self.model = self._tune_model(features, labels, params_grid, cv_iterations, cv_folds)
        else:
            self.model = RandomForestRegressor(random_state=0, **kwargs).fit(X=features, y=labels)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Generates model predictions, given features.
        """
        if self.model is None:
            self._log.error('The model has not been fit.')
        return pd.Series(self.model.predict(features))

    def evaluate(self, labels: pd.Series, predictions: pd.Series) -> Dict[str, str]:
        """
        Evaluates model, comparing predicted and actual values of target variable.
        Computes RMSE, MAE, R-Squared and % predictions greater or less than 600 (seconds) of actual value.
        """
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        mae = mean_absolute_error(labels, predictions)
        r_squared = r2_score(labels, predictions)
        proportion_ten_mins_early = (predictions - labels < -60 * 10).mean()
        proportion_ten_mins_late = (predictions - labels > 60 * 10).mean()

        _process = lambda x: str(round(x, 2))

        evaluation_results = \
            {'Mean Absolute Error': _process(mae), 'R-Squared': _process(r_squared),
             '% >10 mins early': _process(proportion_ten_mins_early),
             '% >10 mins late': _process(proportion_ten_mins_late)}
        self._log.info(evaluation_results)

        return evaluation_results

    def feature_importance(self, feature_names: List[str], plot=False):
        """
        Extracts RandomForestRegressor feature importances.

        plot: (bool, optional)
            If True, plots bar chart of feature importances in descending order.
            If False, returns feature importances as a Series.
        """
        if self.model is None or self.feature_names is None:
            self._log.error('The model has not been fit.')

        feature_names_cleaned = [feature.replace('_', ' ').title() for feature in feature_names]

        # Generate a dict of feature mapping to its importance, in descending order.
        importances = {feature: importance for feature, importance in
                       zip(feature_names_cleaned, self.model.feature_importances_)}
        importances = pd.Series(importances).sort_values(ascending=False)

        if not plot:
            return importances

        sns.barplot(x=importances.values, y=importances.index, color='g')
        plt.xlabel('Feature Importance\n(scaled MSE Reduction over feature splits)')
        plt.ylabel('')
        plt.title('Random Forest Feature Importances')
        plt.show()

    @staticmethod
    def one_hot_encode(data: pd.DataFrame, categorical_variables: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Merges `data` with a one-hot-encoded representation of the columns in `categorical_variables`.

        categorical_variables (dict): dict mapping column to categories, e.g. {'col1': ['L1', 'L2']}
        """
        columns = list(categorical_variables.keys())
        categories = list(categorical_variables.values())

        # Generate a DataFrame of one-hot-encoded columns in the order of `categories`.
        encoder = OneHotEncoder(categories=categories, sparse=False)
        categorical_data_encoded = pd.DataFrame(encoder.fit_transform(data[columns]))

        # Extracts feature name from the column being encoded and the category, e.g. 'country': 'UK' -> 'country_UK'
        categories_column_names = []
        for column, categories in zip(columns, categories):
            for category in categories:
                categories_column_names.append(column + '_' + category)
        categorical_data_encoded.columns = categories_column_names

        # Returns the original data concatenated with the one-hot-encoded data.
        return pd.concat([data, categorical_data_encoded], axis=1)

    @staticmethod
    def plot_residuals_against_predictions(labels: pd.Series, predictions: pd.Series):
        """
        Plots scatterplot of (labels - predictions) against predictions, to identify patterns of residuals as
        predictions increases.
        """
        sns.regplot(x=predictions, y=(labels - predictions), lowess=True, scatter=False, ci=False)
        plt.xlabel('Predicted Food Prep Time / s')
        plt.ylabel('(Actual - Predicted) Prep Time / s')
        plt.title('(Actual - Predicted) against Predicted Food Prep Times')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlim(300, 5000)
        plt.ylim(-600, 1500)
        plt.show()

    @staticmethod
    def mean_impute_nan_values(data_to_impute: pd.DataFrame, data_to_use: pd.DataFrame, columns: List[str]):
        """
        Imputes nan values in the `columns` of `data_to_impute` with mean values in `data_to_use`.
        """
        imputed_data = data_to_impute
        for column in columns:
            imputed_data = imputed_data.fillna({column: data_to_use[column].drop_duplicates().mean()})
        return imputed_data

    @staticmethod
    def _create_params_grid() -> Dict[str, list]:
        """
        Creates a grid of hyperparameters for RandomForestRegressor.
        """
        return {'n_estimators': [int(x) for x in np.linspace(start=100, stop=300, num=5)],
                'max_depth': [int(x) for x in np.linspace(10, 50, num=4)] + [None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['auto', 'sqrt']}

    @staticmethod
    def _tune_model(features: pd.DataFrame, labels: pd.Series, params_grid: Dict[str, list], cv_iterations: int,
                    cv_folds: int) -> RandomForestRegressor:
        """
        Performs Random Search CV on `params_grid` to find optimal hyperparameters for RandomForestRegressor.
        """
        random_search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=params_grid,
                                           n_iter=cv_iterations, cv=cv_folds, n_jobs=-1, verbose=2)
        random_search.fit(X=features, y=labels)

        # Returns the fitted model with optimal hyperparameters from Random Search CV.
        best_model = random_search.best_estimator_
        return best_model
