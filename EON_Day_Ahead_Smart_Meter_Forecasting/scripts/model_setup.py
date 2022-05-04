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
            ys.append(y.iloc[i + self.input_width + self.label_width - 1])
        return {"x": np.array(Xs), "y": np.array(ys)}

    def make_dataset(self, data):
        # y = data.pop("value")
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
                 num_features=4):
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
            window_copy = dataset_to_adjust[window].copy()
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

    def compile_and_fit(self, model, train_data, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        """model.compile(loss=tf.losses.MeanAbsolutePercentageError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])"""

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsolutePercentageError()])

        # train = window.train

        history = model.fit(x=train_data["x"],
                            y=train_data["y"],
                            epochs=self.max_epochs,
                            validation_split=0.1,
                            callbacks=[early_stopping])

        return history

    def train_model(self, data=None, load_model=False, save_model=True):

        multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32,
                                 return_sequences=False,
                                 input_shape=(self.n_before, self.num_features)),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(self.n_ahead * self.num_features,
                                  kernel_initializer=tf.initializers.zeros())
            # tf.keras.layers.Reshape([int(self.n_ahead), self.num_features])
        ])

        if data is not None:
            all_windows = []
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

                    all_windows.append(multi_window)

            test, train, val = self.combine_windows(all_windows)

            with open(f'{settings.DIR_DATA}test_train_val_data_{self.dataset_name}.pkl', 'wb') as f:
                pickle.dump((test, train, val), f)
        else:
            with open(f'{settings.DIR_DATA}test_train_val_data_{self.dataset_name}.pkl', 'rb') as f:
                test, train, val = pickle.load(f)

        history = self.compile_and_fit(multi_lstm_model, train)

        # summarize history for accuracy
        """plt.plot(history.history["mean_absolute_error"])
        plt.plot(history.history["val_mean_absolute_error"])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()"""

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
        self.display_accuracy(train, multi_lstm_model)

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

    def display_accuracy(self, testset, model):
        # Predict things
        x_test = testset["x"]
        y_test = testset["y"]

        predict_input = None
        first = True
        for package in testset["x"]:
            if first:
                first = False
                predict_input = package
            else:
                a = package[-1][:]
                predict_input = np.vstack((predict_input, a))

        x_startset = predict_input[:24, :]

        y_pred = model.predict(x_test)
        y_pred_t = np.transpose(y_pred)
        col = y_pred_t[0]
        plt.plot(y_test, label="y_test")
        plt.show()
        plt.plot(col, label="y_pred")
        plt.show()
        #pass