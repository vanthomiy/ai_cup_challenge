import pickle
import warnings
from enum import Enum

# import IPython
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
        self.train_df = data["train"][0]
        self.val_df = data["validation"][0]
        self.test_df = data["test"][0]

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = int(input_width)
        self.label_width = int(label_width)
        self.shift = int(shift)

        self.total_window_size = int(input_width + shift)

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


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
        self.dataset_time_interval = int(dataset_time_interval)
        '''Sets the interval of the data from the dataset by minutes'''
        self.dataset_time_adjustment = dataset_time_adjustment
        '''Defines how the data is adjusted to other time interval'''
        self.forecast_next_n_minutes = int(forecast_next_n_minutes)
        '''Defines in which interval in minutes tha data should be forecasted (n_ahead is calculated from it)'''
        self.n_ahead = int(forecast_next_n_minutes / dataset_time_interval)
        '''Defines how many data points should be predicted by one prediction'''
        self.previous_data_for_forecast = int(previous_data_for_forecast)
        self.n_before = int(self.n_ahead * previous_data_for_forecast)
        '''Defines how much previous time should be taken in consideration. N times the forecast time'''
        self.train_val_test = train_val_test
        '''Value between 0 and 1 defining the data split (0 = 0% test, )'''
        self.max_epochs = int(max_epochs)
        self.num_features = int(num_features)
        self.model_name = "t_" + str(self.dataset_time_interval) + \
                          "f_" + str(self.forecast_next_n_minutes) + \
                          "pd_" + str(self.previous_data_for_forecast) + \
                          "e_" + str(self.max_epochs)
        self.dataset_name = f"{self.dataset_time_interval}_{self.dataset_time_adjustment}"

    def do_murks(self, dataset):

        dataset_normed = {}

        for sets in ["train", "validation", "test"]:
            dataset_normed[sets] = []
            for data in dataset:
                dataset_normed[sets].append(data[sets])
        return dataset_normed

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
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=self.max_epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

    def train_model(self, data, load_model=False, save_model=True):

        multi_window = WindowGenerator(input_width=self.n_before,
                                       label_width=self.n_ahead,
                                       shift=self.n_ahead,
                                       data=data)

        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            # tf.keras.layers.Flatten(),

            tf.keras.layers.LSTM(128*2, return_sequences=False), #32
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(self.n_ahead * self.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            #tf.keras.layers.Reshape([self.n_ahead, self.num_features])
            tf.keras.layers.Reshape([self.n_ahead, 66])
        ])

        history = self.compile_and_fit(multi_lstm_model, multi_window)

        multi_val_performance = multi_lstm_model.evaluate(multi_window.val)
        multi_performance = multi_lstm_model.evaluate(multi_window.test, verbose=0)
        multi_window.plot(multi_lstm_model)

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # multi_val_performance = multi_lstm_model.evaluate(multi_window.val["x"], multi_window.val["y"])
        # multi_performance = multi_lstm_model.evaluate(multi_window.test["x"], multi_window.test["y"], verbose=0)

        multi_lstm_model.save_weights(settings.DIR_MODEL + self.model_name + '.h5')

    def combine_windows(self, lst_windows: list[WindowGenerator]) -> (dict, dict, dict):
        """
        This function combines the test, train and validation datasets which are generated as an instance of the WindowGenerator.

        :param lst_windows: A list which contains all WindowGenerator instances which were generated from the dataset.
        :return: A tuple consisting of three dictionaries which contain information about the test, train and validation dataset.
        """

        test = {}
        train = {}
        val = {}
        first = True
        for window in lst_windows:
            if first:
                first = False
                test['x'] = window.test['x']
                train['x'] = window.train['x']
                val['x'] = window.val['x']
                test['y'] = window.test['y']
                train['y'] = window.train['y']
                val['y'] = window.val['y']
            else:
                test['x'] = np.vstack((test['x'], window.test['x']))
                train['x'] = np.vstack((train['x'], window.train['x']))
                val['x'] = np.vstack((val['x'], window.val['x']))
                test['y'] = np.hstack((test['y'], window.test['y']))
                train['y'] = np.hstack((train['y'], window.train['y']))
                val['y'] = np.hstack((val['y'], window.val['y']))
        return test, train, val

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
