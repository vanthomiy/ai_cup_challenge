import warnings
from enum import Enum

import IPython
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.common import SettingWithCopyWarning

import settings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class Adjust(Enum):
    CUT = 1
    """Take the values from the particular time"""
    AGGREGATE = 2
    """Sum up the values over the particular timespan"""


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 data,
                 label_columns=None):
        # Store the raw data.
        self.train_df = data["train"]
        self.val_df = data["validation"]
        self.test_df = data["test"]

        # Work out the window parameters.
        self.input_width = int(input_width)
        self.label_width = int(label_width)
        self.shift = int(shift)

        self.total_window_size = input_width + shift

    def create_dataset(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.input_width):
            v = X.iloc[i:(i + self.input_width)].values
            Xs.append(v)
            ys.append(y.iloc[i + self.input_width])
        return {"x": np.array(Xs), "y": np.array(ys)}

    def make_dataset(self, data):
        #y = data.pop("value")
        y = data["value"]
        x = data

        return self.create_dataset(x, y)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


class ModelSetup:
    def __init__(self,
                 dataset_time_interval=30,
                 dataset_time_adjustment=Adjust.CUT,
                 forecast_next_n_minutes=60,
                 previous_data_for_forecast=3,
                 train_val_test=[0.7, 0.2, 0.1],
                 max_epochs=100,
                 num_features=6):
        """
        The actual model setup Defines how the data is processed Defines how the prediction is done
        :param dataset_time_interval: Sets the interval of the data from the dataset by minutes
        :param dataset_time_adjustment: Defines how the data is adjusted to other time interval
        :param forecast_next_n_minutes: Defines how many data points should be predicted by one prediction
        :param previous_data_for_forecast: Defines how much previous time should be taken in consideration. N times the
        :param validation_training_split: Value between 0 and 1 defining the data split (0 = 0% test, )
        forecast time
        """
        self.dataset_time_interval = dataset_time_interval
        '''Sets the interval of the data from the dataset by minutes'''
        self.dataset_time_adjustment = dataset_time_adjustment
        '''Defines how the data is adjusted to other time interval'''
        self.forecast_next_n_minutes = forecast_next_n_minutes
        '''Defines in which interval in minutes tha data should be forecasted (n_ahead is calculated from it)'''
        self.n_ahead = forecast_next_n_minutes / dataset_time_interval
        '''Defines how many data points should be predicted by one prediction'''
        self.previous_data_for_forecast = previous_data_for_forecast
        self.n_before = int(self.n_ahead * previous_data_for_forecast)
        '''Defines how much previous time should be taken in consideration. N times the forecast time'''
        self.train_val_test = train_val_test
        '''Value between 0 and 1 defining the data split (0 = 0% test, )'''
        self.max_epochs = max_epochs
        self.num_features = num_features
        self.model_name = "t_" + str(self.dataset_time_interval) + \
                          "f_" + str(self.forecast_next_n_minutes) + \
                          "pd_" + str(self.previous_data_for_forecast) + \
                          "e_" + str(self.max_epochs)

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
                df_list.append(
                    window_copy[window_copy.index % time_factor == 0])  # Selects every 3rd raw starting from 0
            elif self.dataset_time_adjustment == Adjust.AGGREGATE:
                for i in range(0, len(window_copy.index), time_factor):
                    for pseudo_id in settings.pseudo_ids:
                        values = window_copy[pseudo_id].values
                        total = sum(values[i:i + time_factor])
                        window_copy[pseudo_id][i] = total

                df_list.append(
                    window_copy[window_copy.index % time_factor == 0])  # Selects every 3rd raw starting from 0

        return df_list

    def create_norm_features_data_set(self, dataset):
        """
        Adjust the dataset by the setup parameters
        :param dataset: The actual dataset
        :return The dataset adjusted by the given parameters for the [ModelSetup]
        """
        all_train_data = []
        for data in dataset:
            all_train_data.append(data["train"])
        df = pd.concat(all_train_data)

        train_normed = {}

        train_normed["mean"] = df.mean()
        train_normed["std"] = df.std()

        dataset_normed = {"train": [(data["train"] - train_normed["mean"]) / train_normed["std"] for data in dataset],
                          "validation": [(data["validation"] - train_normed["mean"]) / train_normed["std"] for data in
                                         dataset],
                          "test": [(data["test"] - train_normed["mean"]) / train_normed["std"] for data in dataset]}

        features = settings.features.copy()
        features.extend(settings.pseudo_ids)
        for sets in ["train", "validation", "test"]:
            for i in range(0, len(dataset_normed[sets])):
                dataset_normed[sets][i] = dataset_normed[sets][i][features]

        return dataset_normed, train_normed

    def split_train_validation_test_data(self, dataset_to_split):
        """
        Adjust the dataset by the setup parameters
        :param dataset_to_split: The actual dataset
        :return (test,train) Returns the test and the train dataset
        """

        list = []

        for window in dataset_to_split:
            data_points = len(window.index)
            train_data_points = int(data_points * self.train_val_test[0])
            validate_data_points = int(data_points * self.train_val_test[1])
            list.append({"train": window[:train_data_points],
                         "validation": window[train_data_points:train_data_points + validate_data_points],
                         "test": window[train_data_points + validate_data_points:]})

        return list

    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

        train = window.train

        history = model.fit(x=window.train["x"],
                            y=window.train["y"],
                            epochs=self.max_epochs,
                            validation_split=0.1,
                            callbacks=[early_stopping])

        """history = model.fit(train_x, train_y, epochs=self.max_epochs,
                            callbacks=[early_stopping])"""

        return history

    def train_model(self, data, load_model=False, save_model=True):

        multi_lstm_model = tf.keras.Sequential()
        multi_lstm_model.add(tf.keras.layers.LSTM(
            units=128,
            input_shape=(self.n_before, self.num_features)
        ))
        multi_lstm_model.add(tf.keras.layers.Dense(units=1))
        multi_lstm_model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(0.001)
        )

        """if load_model:
            multi_lstm_model = load_model
        else:
            multi_lstm_model = tf.keras.Sequential([
                # Shape [batch, time, features] => [batch, lstm_units].
                # Adding more `lstm_units` just overfits more quickly.
                tf.keras.layers.LSTM(32, return_sequences=False),
                # Shape => [batch, out_steps*features].
                tf.keras.layers.Dense(self.n_ahead * self.num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features].
                tf.keras.layers.Reshape([self.n_ahead, self.num_features])
            ])"""

        for pseudo_id in settings.pseudo_ids:
            data_for_id = []
            headers = [pseudo_id]
            headers.extend(settings.features)

            for i in range(0, len(data["train"])):
                data_dict = {}
                for value in ["train", "validation", "test"]:
                    copy_df = data[value][i][headers]
                    copy_df.rename(columns={pseudo_id: 'value'}, inplace=True)
                    # we use 0 to x for the pseudo ids...
                    copy_df["pseudo_id"] = settings.pseudo_ids.index(pseudo_id)
                    data_dict[value] = copy_df
                data_for_id.append(data_dict)

            for window_data_for_id in data_for_id:
                multi_window = WindowGenerator(input_width=self.n_before,
                                               label_width=self.n_ahead,
                                               shift=self.n_ahead,
                                               data=window_data_for_id)

                history = self.compile_and_fit(multi_lstm_model, multi_window)

                multi_val_performance = multi_lstm_model.evaluate(multi_window.val["x"], multi_window.val["y"])
                multi_performance = multi_lstm_model.evaluate(multi_window.test["x"], multi_window.test["y"], verbose=0)

        multi_lstm_model.save_weights(settings.DIR_MODEL + self.model_name + '.h5')

    def load_model(self):
        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(self.n_ahead * self.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([self.n_ahead, self.num_features])
        ])

        multi_lstm_model = multi_lstm_model.load_weights(settings.DIR_MODEL + self.model_name + '.h5')

        return multi_lstm_model

    def test_model(self, save_metrics=True):
        pass

    def display_metrics(self, validation):
        pass

    def display_dataset(self, dataset):
        pass
