from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import settings


class DatasetHandler():
    def __init__(self, path):
        self.path = path
        self.train_df = pd.read_csv(path, index_col='pseudo_id')
        self.time_based_df = self.time_based_dataset()
        self.features_df = self.create_features()

    def time_based_dataset(self):
        """
        We create a dictionary for each pseudo id
        In this dictionary there is a list of the datasets. For each 45-day window one list
        """
        transponsed_df = self.train_df.T


        # for each column (id) in the df

        split_by = 38 * 24 * 2

        # loop through the data and split by any break between the timestamps
        list_of_dfs = []

        for i in range(0, int(len(transponsed_df.index) / split_by)):
            list_of_dfs.append(transponsed_df.iloc[i * split_by:(i + 1) * split_by])

        return list_of_dfs

    def create_features(self):

        data = {}

        day = 24 * 60 * 60
        year = (365.2425) * day
        window_id = 0

        for window in self.time_based_df:
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
