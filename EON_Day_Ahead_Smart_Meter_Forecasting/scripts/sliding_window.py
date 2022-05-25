import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.setup import ID_HANDLING


class Windowing:
    def __init__(self, setup):
        self.setup = setup

    def load_time_window_data(self):
        train_dfs = []
        test_dfs = []
        val_dfs = []

        for time in range(0, self.setup.ACTUAL_SETUP.time_windows_to_use):
            df = pd.read_csv(self.setup.FILE_TIME_WINDOW_X(time))
            for pseudo_id in self.setup.PSEUDO_IDS:
                features = [pseudo_id]
                features.extend(self.setup.ACTUAL_SETUP.features)
                features.extend(self.setup.ACTUAL_SETUP.weather_features)
                df_id = df[features]
                df_id.rename(columns={pseudo_id: 'value'}, inplace=True)
                features = ["value"]
                features.extend(self.setup.ACTUAL_SETUP.features)
                features.extend(self.setup.ACTUAL_SETUP.weather_features)

                if self.setup.ACTUAL_SETUP.id_handling == ID_HANDLING.MULTIPLE:
                    df_id["pseudo_id"] = self.setup.PSEUDO_IDS.index(pseudo_id)

                # df is list of all values
                n = len(df_id)
                train_dfs.append(df_id[0:int(n * 0.6)])
                # val_dfs.append(df_id[int(n * 0.6):int(n * 0.8)])
                val_dfs.append(df_id[int(n * 0.6):-7 * 24])
                # test_dfs.append(df_id[int(n * 0.8):])
                test_dfs.append(df_id[-7 * 24:])

        return {"train": train_dfs, "test": test_dfs, "val": val_dfs}

    def split_window(self, features):
        total_window_size = self.setup.ACTUAL_SETUP.n_before + self.setup.ACTUAL_SETUP.n_ahead
        input_slice = slice(0, self.setup.ACTUAL_SETUP.n_before)

        label_start = total_window_size - self.setup.ACTUAL_SETUP.n_ahead
        labels_slice = slice(label_start, None)

        inputs = features[:, input_slice, :]
        labels = features[:, labels_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.setup.ACTUAL_SETUP.n_before, None])
        labels.set_shape([None, self.setup.ACTUAL_SETUP.n_ahead, None])

        return inputs, labels

    def make_dataset(self, data_list):
        ds = None
        total_window_size = self.setup.ACTUAL_SETUP.n_before + self.setup.ACTUAL_SETUP.n_ahead

        for data in data_list:
            data = np.array(data, dtype=np.float32)
            ds_temp = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32)

            ds_temp = ds_temp.map(self.split_window)
            if ds is None:
                ds = ds_temp
            else:
                ds = ds.concatenate(ds_temp)

        # We need to create a list out of it to store it with pkl
        # result = ds.unbatch()
        # res1 = list(result)

        return ds

    def create_window(self, data_dict, id=-1):
        # windowed dict
        windowed_data = {}

        # create windowed data for train test and validation
        for value in self.setup.TEST_TRAIN_VALID:
            data = self.make_dataset(data_dict[value])

            tf.data.experimental.save(
                data, self.setup.FILE_WINDOWED_DATA(value, (str(id) if id > -1 else "all")))

    def start(self):
        # load the data
        data_dict = self.load_time_window_data()

        if self.setup.ACTUAL_SETUP.id_handling == ID_HANDLING.SINGLE:
            for id in range(0, self.setup.ACTUAL_SETUP.pseudo_id_to_use):
                id_df = {}
                for key in self.setup.TEST_TRAIN_VALID:
                    id_df[key] = data_dict[key][id::self.setup.ACTUAL_SETUP.pseudo_id_to_use]
                self.create_window(id_df, id)
        else:
            self.create_window(data_dict)


