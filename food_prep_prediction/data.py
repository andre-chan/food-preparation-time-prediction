"""
Helper functions for preprocessing data.
"""

from datetime import datetime
from typing import Tuple, List

import pandas as pd
import os
import logging

import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()


def read_data(project_filepath: str = os.getcwd()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads 'orders' and 'restaurants' tables from '{project_filepath}/data/*.csv'.
    project_filepath: Absolute path to the project directory
    """
    orders = pd.read_csv(os.path.join(project_filepath, 'data', 'orders.csv.gz'))
    restaurants = pd.read_csv(os.path.join(project_filepath, 'data', 'restaurants.csv.gz'))
    return orders, restaurants


def train_test_split_by_date(data: pd.DataFrame, split_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the data into datasets for train (orders acknowledged on or before 24/06) & test (orders after 25/06).
    """
    train_data = data.loc[data['order_acknowledged_at_local'] <= split_date]
    test_data = data.loc[data['order_acknowledged_at_local'] > split_date]
    return train_data, test_data


def convert_str_to_timestamp(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Converts the columns in data consisting of dates as strings, to dates as pd.Timestamp objects.
    """
    for column in columns:
        data[column] = pd.to_datetime(data[column])
        data[column + '_local'] = data[column].apply(lambda date: date.replace(tzinfo=None))  # remove timezone info
    return data


