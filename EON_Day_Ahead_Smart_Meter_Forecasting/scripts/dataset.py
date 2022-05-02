import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


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

        data = {}

        # for each column (id) in the df
        for column in transponsed_df:
            data[column] = []
            df_for_id = transponsed_df[[column]].copy()
            df_for_id.rename(
                columns={column: "value"}, inplace=True)
            df_for_id["pseudo_id"] = [column] * len(df_for_id.index)

            split_by = 38 * 24 * 2

            # loop through the data and split by any break between the timestamps
            list_of_dfs = []

            for i in range(0, int(len(df_for_id.index) / split_by)):
                list_of_dfs.append(df_for_id.iloc[i * split_by:(i + 1) * split_by])
            data[column] = (list_of_dfs)

        return data

    def create_features(self):

        data = {}

        day = 24 * 60 * 60
        year = (365.2425) * day

        for pseudo_id in self.time_based_df:
            data[pseudo_id] = []
            for window in self.time_based_df[pseudo_id]:
                df = pd.DataFrame("value", "pseudo_id", "day sin", "day cos", "year sin", "year cos")

                for row, index in window.itterrows():
                    timestamp_s = index.map(pd.Timestamp.timestamp)

                    df["value"] = row["value"]
                    df["pseudo_id"] = row["pseudo_id"]
                    df["day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
                    df["day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
                    df['year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
                    df['year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
                data[pseudo_id].append(df)

        return data