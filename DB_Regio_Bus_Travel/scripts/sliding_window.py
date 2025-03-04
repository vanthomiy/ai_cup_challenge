
import numpy as np
import pandas as pd
import tensorflow as tf

import settings


def load_time_window_data():
    train_dfs = []
    test_dfs = []
    val_dfs = []

    for time in range(0, settings.ACTUAL_SETUP.time_windows_to_use):
        df = pd.read_csv(settings.FILE_TIME_WINDOW_X(time))
        for pseudo_id in settings.BUS_STOPS_SORTED:
            pseudo_id = str(pseudo_id)
            if df[pseudo_id].isnull().values.any():
                continue

            features = [pseudo_id]
            features.extend(settings.ACTUAL_SETUP.features)
            features.extend(settings.ACTUAL_SETUP.weather_features)
            df_id = df[features]
            df_id.rename(columns={pseudo_id: 'value'}, inplace=True)
            features = ["value"]
            features.extend(settings.ACTUAL_SETUP.features)
            features.extend(settings.ACTUAL_SETUP.weather_features)
            try:
                df_id["pseudo_id"] = settings.BUS_STOPS_SORTED.index((pseudo_id))
            except Exception as ex:
                pass

            # df is list of all values
            n = len(df_id)
            """train_dfs.append(df_id[0:int(7*24*1)])
            val_dfs.append(df_id[int(7*24*1):(7*24*2)])
            test_dfs.append(df_id[(7*24*2):])"""
            train_dfs.append(df_id[0:int(n*0.33)])
            val_dfs.append(df_id[int(n*0.33):int(n*0.66)])
            test_dfs.append(df_id[int(n*0.66):])

    return {"train": train_dfs, "test": test_dfs, "val": val_dfs}


def split_window(features):
    total_window_size = settings.ACTUAL_SETUP.n_before + settings.ACTUAL_SETUP.n_ahead
    input_slice = slice(0, settings.ACTUAL_SETUP.n_before)

    label_start = total_window_size - settings.ACTUAL_SETUP.n_ahead
    labels_slice = slice(label_start, None)

    inputs = features[:, input_slice, :]
    labels = features[:, labels_slice, :]

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, settings.ACTUAL_SETUP.n_before, None])
    labels.set_shape([None, settings.ACTUAL_SETUP.n_ahead, None])

    return inputs, labels


def make_dataset(data_list):
    ds = None
    total_window_size = settings.ACTUAL_SETUP.n_before + settings.ACTUAL_SETUP.n_ahead

    for data in data_list:
        try:
            data = np.array(data, dtype=np.float32)
            ds_temp = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32)

            ds_temp = ds_temp.map(split_window)
            if ds is None:
                ds = ds_temp
            else:
                ds = ds.concatenate(ds_temp)
        except Exception:
            pass

    # We need to create a list out of it to store it with pkl
    # result = ds.unbatch()
    # res1 = list(result)

    return ds


# load the data
data_dict = load_time_window_data()

# windowed dict
windowed_data = {}

# create windowed data for train test and validation
for value in settings.TEST_TRAIN_VALID:
    data = make_dataset(data_dict[value])

    tf.data.experimental.save(
        data, settings.FILE_WINDOWED_DATA(value))
