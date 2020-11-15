"""
Plotting functions for data exploration and visualisation.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger()


def plot_prep_time_vs_no_items_ordered(train_data: pd.DataFrame):
    """
    Plots `pointplot` of food prep time against number of items ordered.
    """
    # Bin number of items for visualisation only, but not for passing to the model.
    number_of_items_binned = pd.cut(train_data['number_of_items'],
                                    bins=[0, 5, 10, 15, 20, 100],
                                    labels=['0 - 5', '6 - 10', '11 - 15', '16 - 20', '20+'])
    data = pd.DataFrame(
        {'prep_time_seconds': train_data.prep_time_seconds, 'number_of_items_binned': number_of_items_binned})
    ax = sns.pointplot(x='number_of_items_binned', y='prep_time_seconds', data=data, join=False)
    ax.set(xlabel='Number of Items Ordered', ylabel='Prep Time / s',
           title='Average Food Prep Time by Number of Items Ordered')
    plt.show()


def plot_prep_time_vs_meal_occasion(train_data: pd.DataFrame):
    """
    Plots `pointplot` of food prep time against meal occasion ordered.
    """
    ax = sns.pointplot(x='meal_occasion', y='prep_time_seconds', data=train_data,
                       order=['lunch', 'early_night', 'late_night'], join=False)
    ax.set(xlabel='Meal Occasion', ylabel='Prep Time / s',
           title='Average Food Prep Time by Meal Occasion')
    plt.show()


def plot_prep_time_vs_city(train_data: pd.DataFrame):
    """
    Plots `pointplot` of food prep time against city.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax = sns.pointplot(x='city', y='prep_time_seconds', data=train_data, join=False)
    ax.set(xlabel='City', ylabel='Prep Time / s',
           title='Average Food Prep Time by City')
    plt.show()


def plot_prep_time_vs_type_of_food(train_data: pd.DataFrame):
    """
    Plots `pointplot` of food prep time against type of food.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax = sns.pointplot(x='type_of_food', y='prep_time_seconds', data=train_data, join=False)
    ax.set(xlabel='Type of Food  (names omitted)', ylabel='Prep Time / s',
           title='Average Food Prep Time by Type of Food')
    ax.set_xticks([])
    plt.show()


def get_high_pairwise_correlated_features(train_data: pd.DataFrame, threshold: float = 0.3):
    """
    Logs distinct pairs of features with Pearson correlation exceeding `threshold`.
    """
    pairwise_correlations = train_data.drop(['restaurant_id', 'prep_time_seconds'], axis=1).corr().abs()
    # Rank pairwise (absolute) correlations in descending order
    ranked_pairwise_correlations = round(pairwise_correlations, 2).unstack().sort_values(ascending=False)

    # Filter correlations to exceed `threshold` and select distinct feature pairs
    filtered_correlations = ranked_pairwise_correlations[
                                (ranked_pairwise_correlations > threshold) & (ranked_pairwise_correlations < 1)][1::2]
    logger.info('Highest Pairwise Feature Correlations\n')
    logger.info(filtered_correlations.to_string())


def long_food_prep_time_stats(data: pd.DataFrame):
    """
    Logs the number and proportion of orders taking more than 3 hours to prepare, and logs the highest from the dataset.
    """
    count_more_than_three_hours_prep_time = np.array(data['prep_time_seconds'] >= 3600 * 3).sum()
    max_prep_time = round(data['prep_time_seconds'].max() / 60, 1)
    logger.info(f'{count_more_than_three_hours_prep_time} orders '
                f'({round(count_more_than_three_hours_prep_time / len(data), 2) * 100}%) took more '
                f'than 3 hours of food prep time. The longest took {max_prep_time} hours.')
