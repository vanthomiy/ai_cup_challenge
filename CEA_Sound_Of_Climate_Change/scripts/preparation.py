import pickle
from datetime import datetime

import librosa
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import settings as set
import pandas as pd
from scripts.setup import Normalization, Timespan
from scipy.io import loadmat


def array_to_number(arr):
    for i in range(0, len(arr)):
        if arr[i] == 1:
            return i


class Preparation:
    def __init__(self, setup):
        self.setup = setup

    def load_data(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Load data from the 'train.csv' and 'counts.csv' csv files.

        :return: Two DataFrames which contain the data.
        """
        annots_train = loadmat(self.setup.FILE_TRAIN_DATA)
        annots = loadmat(self.setup.FILE_TEST_DATA)
        predict_train = [[element for element in upperElement] for upperElement in annots_train['noisy_SignalDATA']]
        labels_train = [[element for element in upperElement] for upperElement in annots_train['ClassDATA']]
        validation = [[element for element in upperElement] for upperElement in annots['noisy_SignalDATA']]

        df1 = pd.DataFrame(predict_train).T
        df2 = pd.DataFrame(labels_train)

        # zip provides us with both the x and y in a tuple.
        df3 = pd.DataFrame(validation).T

        return df1, df2, df3

    def normalize_data_NONE(self):
        """
        Normalization algorithm which is to be defined.

        :return: tbd.
        """
        raise NotImplemented("Not Implemented yet.")

    def normalize_data_MEAN(self, train_transposed: pd.DataFrame, test_df) -> (pd.DataFrame, dict):
        """
        Normalization which uses the MEAN algorithm.

        :param train_transposed: The train_cleaned dataset as a DataFrame, provided by the function clean_train_dataset.
        :return: df: The train_cleaned dataset as a copy and normalized.
                 _normalization: The used normalization values as a dictionary.
        """

        df = train_transposed.copy()

        # Calculate the normalization values and save them in a dictionary
        _normalization = {
            "mean": df.to_numpy().mean(),
            "std": df.to_numpy().std()
        }
        df = (df - _normalization["mean"]) / _normalization["std"]
        test_df = (test_df - _normalization["mean"]) / _normalization["std"]
        return df, _normalization, test_df

    def normalize_data_ZERO_TO_ONE(self, train_transposed: pd.DataFrame, test_df) -> (pd.DataFrame, dict):
        """
        Normalization which uses the ZERO TO ONE algorithm.

        :param train_transposed: The train_cleaned dataset as a DataFrame, provided by the function clean_train_dataset.
        :return: df: The train_cleaned dataset as a copy and normalized.
                 _normalization: The used normalization values as a dictionary.
        """

        df = train_transposed.copy()
        # Calculate the normalization values and save them in a dictionary
        _normalization = {
            "max": df.to_numpy().max(),
            "min": df.to_numpy().min()
        }
        df = (df - _normalization["min"]) / _normalization["max"]
        test_df = (test_df - _normalization["mean"]) / _normalization["std"]
        return df, _normalization, test_df

    def save_normalization_values(self, normalization_values: dict):
        """
        Save the normalization values as pickle file.
        :param normalization_values:
        """

        with open(self.setup.FILE_NORMALIZATION_DATA, 'wb') as file:
            pickle.dump(normalization_values, file)

    def create_and_save_data_windows(self, d1_data, d2_labels, d3_test):
        """
        Create and save data windows with sin and cos values for both daily and annual ranges.

        :param lst_split_dataset: A list containing the split dataset provided by the function split_dataset.
        :param _amplitude: The amplitude used for the sin and cos functions.
        :param _offset: The offset used for the sin and cos functions.
        """

        labels = [array_to_number(row) for index, row in d2_labels.iterrows()]

        # Create a dataframe from the list which contains every
        d1_data["labels"] = labels

        # Save the dataframe to a csv file with the current id
        d1_data.to_csv(self.setup.FILE_PRECESSED_DATA_TRAIN)
        d3_test.to_csv(self.setup.FILE_PRECESSED_DATA_TEST)

    def adjust_time_interval(self, df_t):
        """
        Adjusts the time interval of the dataframe.
        For 0-23 Hours-Intervals the data will be cut off
        For 24 Hour-Interval the data will be summed up
        :param df_t: The actual, already transposed, df
        :return: The intervall adjusted df
        """
        count = len(df_t.columns)

        if self.setup.ACTUAL_SETUP.data_interval != Timespan.ALL:
            return df_t
        else:
            take = 0.1
            return df_t.iloc[:, int(count * take):-int(count * take)]


    def save_normalization_plot(self, df_n):
        df_melt = df_n.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_melt)
        _ = ax.set_xticklabels(df_n.keys(), rotation=90)
        plt.savefig(self.setup.FILE_NORMALIZATION_PLOT)

    def start(self):

        # Load the data from the csv files
        train_df, train_df_labels, test_df = self.load_data()

        # Adjust the time intervall of the data [half-hourly, hourly and daily]
        train_df = self.adjust_time_interval(train_df)
        test_df = self.adjust_time_interval(test_df)

        # Introduce necessary variables and predefine them as None
        normalization = None

        # Choose normalization method
        if self.setup.ACTUAL_SETUP.normalization == Normalization.NONE:
            train_df = self.normalize_data_NONE()
        elif self.setup.ACTUAL_SETUP.normalization == Normalization.MEAN:
            train_df, normalization, test_df = self.normalize_data_MEAN(
                train_df, test_df)
        elif self.setup.ACTUAL_SETUP.normalization == Normalization.ZERO_TO_ONE:
            train_df, normalization, test_df = self.normalize_data_ZERO_TO_ONE(
                train_df, test_df)

        # Save normalization values
        self.save_normalization_values(normalization_values=normalization)

        # Save the plot for the normalized data
        self.save_normalization_plot(train_df)

        # Split dataset at the gaps in the dataset

        # Create windows
        self.create_and_save_data_windows(train_df, train_df_labels, test_df)

        print("Preparation script executed successfully.")
