import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

import settings


class DatasetHandler():
    def __init__(self, path):
        self.path = path

    def load_dataset_and_create_features(self):
        """
        We create a dictionary for each pseudo id
        In this dictionary there is a list of the datasets. For each 45-day window one list
        """

        train_df = pd.read_csv(self.path, index_col='pseudo_id')
        path = "../data/start/counts.csv"
        dwellings_count_df = pd.read_csv(path, index_col='pseudo_id')

        # Divide every value in the dataset with the amount of dwellings for this data
        for index, _ in train_df.iterrows():
            factor = dwellings_count_df.loc[index]['n_dwellings']
            train_df.loc[index] = train_df.loc[index].div(factor)

        transponsed_df = train_df.T

        # for each column (id) in the df

        split_by = 38 * 24 * 2

        # loop through the data and split by any break between the timestamps
        list_of_dfs = []

        for i in range(0, int(len(transponsed_df.index) / split_by)):
            list_of_dfs.append(transponsed_df.iloc[i * split_by:(i + 1) * split_by])

        data = {}

        day = 24 * 60 * 60
        year = (365.2425) * day
        window_id = 0

        for window in list_of_dfs:
            dict_list = []
            for index, row in window.iterrows():
                date_time = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
                timestamp_s = int(round(date_time.timestamp()))
                df = {"day sin": np.sin(timestamp_s * (2 * np.pi / day)),
                      "day cos": np.cos(timestamp_s * (2 * np.pi / day)),
                      'year sin': np.sin(timestamp_s * (2 * np.pi / year)),
                      'year cos': np.cos(timestamp_s * (2 * np.pi / year)),
                      'datetime': date_time}

                for pseudo_id in settings.pseudo_ids:
                    df[pseudo_id] = row[pseudo_id]

                dict_list.append(df)

            df = pd.DataFrame(dict_list)
            data[window_id] = df
            df.to_csv(settings.DIR_DATA + "prepared/" + str(window_id) + ".csv")
            window_id += 1
        return data

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