import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


class DatasetHandler():
    def __init__(self, path):
        self.path = path
        self.train_df = pd.read_csv(path, index_col='pseudo_id')
        self.time_based_df = self.time_based_dataset()

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

            size = 38 * 24 * 2
            list_of_dfs = [df_for_id.loc[i:i + size - 1, :] for i in range(0, len(df_for_id), size)]
            data[column].append(list_of_dfs)

        return data