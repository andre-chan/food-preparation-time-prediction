"""
Helper functions for feature engineering.
"""

from datetime import time
from typing import List

import pandas as pd

from category_encoders import TargetEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder

import warnings

warnings.filterwarnings('ignore')


def target_encode(column: pd.Series, labels: pd.Series, leave_one_out=True) -> pd.DataFrame:
    """
    Transform the column to encode the average food prep time for that column.

    leave_one_out (bool): Excludes the current row when calculating the average.
    """
    if leave_one_out:
        # Adds Gaussian noise to training set feature values before target-encoding, to avoid overfitting.
        encoder = LeaveOneOutEncoder(sigma=0.01)
    else:
        encoder = TargetEncoder()
    encoded_column = encoder.fit_transform(X=column, y=labels)
    encoded_column.rename(columns={column.name: column.name + '_mean_prep_time'}, inplace=True)

    # Return a DataFrame with schema ['{column}', '{column}_mean_prep_time']
    return pd.concat([column, encoded_column], axis=1)


def bin_timestamps_to_meal_occasions(column: pd.Series) -> pd.Series:
    """
    Transform column into a 'meal_occasion' variable with levels: [breakfast, lunch, early_night,
    late_night], defined as the intervals: 0600-1100, 1100-1600, 1600-2100, 2100-0600 respectively.
    """
    def get_meal_occasion(timestamp: pd.Timestamp) -> str:
        if not isinstance(timestamp, pd.Timestamp):
            raise ValueError('timestamp must be of type pd.Timestamp')
        if time(hour=6, minute=00) <= timestamp.time() < time(hour=11, minute=00):
            return 'breakfast'
        elif time(hour=11, minute=00) <= timestamp.time() < time(hour=16, minute=00):
            return 'lunch'
        elif time(hour=16, minute=00) <= timestamp.time() < time(hour=21, minute=00):
            return 'early_night'
        else:
            return 'late_night'

    column_meal_occasion = column.apply(lambda timestamp: get_meal_occasion(timestamp))

    return column_meal_occasion


def bin_timestamps_to_weekend_flag(column: pd.Series) -> pd.Series:
    """
    Transform the timestamp column to one-hot-encode a weekend (Fri/Sat/Sun) or not.
    """
    column_is_weekend = column.apply(lambda timestamp: 1 if timestamp.weekday() in [4, 5, 6] else 0)
    return column_is_weekend


def extract_restaurant_feature(grouped_data: pd.DataFrame, restaurant_feature: str) -> pd.DataFrame:
    """
    Extract the following features: restaurant price per item, restaurant average order value, restaurant average
    daily orders (when open).
    """
    if restaurant_feature == 'restaurant_price_per_item':
        func = lambda restaurant_data: restaurant_data.order_value_gbp.sum() / restaurant_data.number_of_items.sum()

    elif restaurant_feature == 'restaurant_average_order_value':
        func = lambda restaurant_data: restaurant_data.order_value_gbp.mean()

    elif restaurant_feature == 'restaurant_average_daily_orders':
        func = lambda restaurant_data: \
            len(restaurant_data) / restaurant_data['order_acknowledged_at_local'].apply(lambda x: x.date()).nunique()

    else:
        raise ValueError(f'Restaurant feature {restaurant_feature} is not supported.')

    # Return a DataFrame with schema ['restaurant_id', '{restaurant_feature}']
    restaurant_feature_with_id = grouped_data.apply(func).reset_index()
    restaurant_feature_with_id.columns = ['restaurant_id', restaurant_feature]

    return restaurant_feature_with_id


def merge_restaurant_features_to_data(data: pd.DataFrame, restaurant_feature_with_id: List[pd.DataFrame]):
    """
    Left joins 'data' with each dataframe in 'restaurant_feature_with_id', on restaurant_id.
    """
    for feature in restaurant_feature_with_id:
        data = data.merge(feature, how='left', on='restaurant_id')
    return data
