import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

import settings


class DatasetHandler:
    def __init__(self, train_data_path, counts_data_path):
        self.train_data_path = train_data_path
        self.counts_data_path = counts_data_path

    def load_dataset_and_create_features(self):
        """
        We create a dictionary for each pseudo id
        In this dictionary there is a list of the datasets. For each 45-day window one list
        """

        # Load data from the corresponding csv files
        train_df = pd.read_csv(self.train_data_path, index_col='pseudo_id')
        counts_df = pd.read_csv(self.counts_data_path, index_col='pseudo_id')

        # Divide every value in the train_df dataset with the amount of dwellings for the value in order to get the
        # average value for each cell
        for index, _ in train_df.iterrows():
            factor = counts_df.loc[index]['n_dwellings']
            train_df.loc[index] = train_df.loc[index].div(factor)

        # Transpose the train_df in order to ToDo Why are we doing this?
        train_transposed_df = train_df.T

        norm_max_min = True  # true is max min
        offset = 0
        amplitude = 2
        normalization = {}

        if norm_max_min:
            normalization["max"] = train_transposed_df.to_numpy().max()
            normalization["min"] = train_transposed_df.to_numpy().min()
            train_transposed_df = (train_transposed_df - normalization["min"]) / normalization["max"]
            offset = 0.5
            amplitude = 0.5
        else:
            normalization["mean"] = train_transposed_df.to_numpy().mean()
            normalization["std"] = train_transposed_df.to_numpy().std()
            train_transposed_df = (train_transposed_df - normalization["std"]) / normalization["mean"]

        # for each column (id) in the df

        split_by = 38 * 24 * 2

        # loop through the data and split by any break between the timestamps
        list_of_dfs = []

        for i in range(0, int(len(train_transposed_df.index) / split_by)):
            list_of_dfs.append(train_transposed_df.iloc[i * split_by:(i + 1) * split_by])

        data = {}

        day = 24 * 60 * 60
        year = (365.2425) * day
        window_id = 0

        for window in list_of_dfs:
            dict_list = []
            for index, row in window.iterrows():
                date_time = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
                timestamp_s = int(round(date_time.timestamp()))
                df = {"day sin": amplitude * np.sin(timestamp_s * (np.pi / day)) + offset,
                      "day cos": amplitude * np.cos(timestamp_s * (np.pi / day)) + offset,
                      'year sin': amplitude * np.sin(timestamp_s * (np.pi / year)) + offset,
                      'year cos': amplitude * np.cos(timestamp_s * (np.pi / year)) + offset}

                for pseudo_id in settings.pseudo_ids:
                    df[pseudo_id] = row[pseudo_id]

                dict_list.append(df)

            df = pd.DataFrame(dict_list)
            data[window_id] = df
            df.to_csv(settings.DIR_DATA + "prepared/" + str(window_id) + ".csv")
            window_id += 1

        with open(f'{settings.DIR_DATA}/normed_values.pkl', 'wb') as f:
            pickle.dump(normalization, f)

        return data

    @staticmethod
    def load_normalization():
        with open(f'{settings.DIR_DATA}/normed_values.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_features_data():
        """
        We create a dictionary for each pseudo id
        In this dictionary there is a list of the datasets. For each 45-day window one list
        """

        data = []

        window_id = 0
        available = True

        while available:
            try:
                data.append(pd.read_csv(settings.DIR_DATA + "prepared/" + str(window_id) + ".csv"))
                window_id += 1
            except:
                available = False

        """ Alternative way
        directory_path = os.fsencode(f"{settings.DIR_DATA}prepared/")
        for file in os.listdir(directory_path):
            filename = os.fsdecode(file)
            data.append(pd.read_csv(filename))"""

        return data

    @staticmethod
    def plot_data(df):
        plot_cols = ['datetime', 'day sin', 'year sin', settings.pseudo_ids[0], settings.pseudo_ids[1]]
        plot_features = df[0][plot_cols]
        _ = plot_features.plot(subplots=True)
        plt.show()
