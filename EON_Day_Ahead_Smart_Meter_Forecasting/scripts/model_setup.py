import warnings
from enum import Enum

import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import settings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class Adjust(Enum):
    CUT = 1
    """Take the values from the particular time"""
    AGGREGATE = 2
    """Sum up the values over the particular timespan"""



class ModelSetup:
    def __init__(self,
                 dataset_time_interval=30,
                 dataset_time_adjustment=Adjust.CUT,
                 forecast_next_n_minutes=60,
                 previous_data_for_forecast=3,
                 test_training_split=0.75):
        """
        The actual model setup Defines how the data is processed Defines how the prediction is done
        :param dataset_time_interval: Sets the interval of the data from the dataset by minutes
        :param dataset_time_adjustment: Defines how the data is adjusted to other time interval
        :param forecast_next_n_minutes: Defines how many data points should be predicted by one prediction
        :param previous_data_for_forecast: Defines how much previous time should be taken in consideration. N times the
        :param test_training_split: Value between 0 and 1 defining the data split (0 = 0% test, )
        forecast time
        """
        self.dataset_time_interval = dataset_time_interval
        '''Sets the interval of the data from the dataset by minutes'''
        self.dataset_time_adjustment = dataset_time_adjustment
        '''Defines how the data is adjusted to other time interval'''
        self.forcast_next_n_minutes = forecast_next_n_minutes
        '''Defines in which interval in minutes tha data should be forecasted (n_ahead is calculated from it)'''
        self.n_ahead = forecast_next_n_minutes / dataset_time_interval
        '''Defines how many data points should be predicted by one prediction'''
        self.previous_data_for_forecast = previous_data_for_forecast
        '''Defines how much previous time should be taken in consideration. N times the forecast time'''
        self.test_training_split = test_training_split
        '''Value between 0 and 1 defining the data split (0 = 0% test, )'''

    def adjust_dataset_time(self, dataset_to_adjust):
        """
        Adjust the dataset by the setup parameters
        :param dataset_to_adjust: The actual dataset
        :return The dataset adjusted by the given parameters for the [ModelSetup]
        """
        # 30 Min = 1, 60 Min = 2, etc.
        time_factor = int(self.dataset_time_interval / 30)
        df_list = []

        for window in dataset_to_adjust:
            window_copy = window.copy()
            if self.dataset_time_adjustment == Adjust.CUT:
                df_list.append(window_copy[window_copy.index % time_factor == 0])  # Selects every 3rd raw starting from 0
            elif self.dataset_time_adjustment == Adjust.AGGREGATE:
                for i in range(0, len(window_copy.index), time_factor):
                    for pseudo_id in settings.pseudo_ids:
                        values = window_copy[pseudo_id].values
                        total = sum(values[i:i+time_factor])
                        window_copy[pseudo_id][i] = total

                df_list.append(window_copy[window_copy.index % time_factor == 0])  # Selects every 3rd raw starting from 0

        return df_list

    def create_norm_features_data_set(self, dataset):
        """
        Adjust the dataset by the setup parameters
        :param dataset: The actual dataset
        :return The dataset adjusted by the given parameters for the [ModelSetup]
        """

    def split_test_train_data(self, dataset_to_split):
        """
        Adjust the dataset by the setup parameters
        :param dataset_to_split: The actual dataset
        :return (test,train) Returns the test and the train dataset
        """
        data_points = dataset_to_split.count()
        test_data_points = int(data_points * self.test_training_split)
        return dataset_to_split[:test_data_points], dataset_to_split[test_data_points:]

    def train_model(self, dataset, save_model=True):
        pass

    def load_model(self):
        pass

    def test_model(self, save_metrics=True):
        pass

    def display_metrics(self, validation):
        pass

    def display_dataset(self, dataset):
        pass
