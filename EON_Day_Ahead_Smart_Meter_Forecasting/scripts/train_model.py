import pickle

import tensorflow as tf
from matplotlib import pyplot as plt

from scripts.setup import ModelParameter, Algorithm, ID_HANDLING


class TrainModel:
    def __init__(self, setup):
        self.setup = setup

    def load_sliding_window(self, id=-1):
        # load the pickle file with the windowed data
        mv_ds = {}

        for value in self.setup.TEST_TRAIN_VALID:
            ds = tf.data.experimental.load(
                self.setup.FILE_WINDOWED_DATA(value, (str(id) if id > -1 else "all")))
            mv_ds[value] = ds

        return mv_ds

    def create_model(self, param):
        act_model = None

        if param.algorithm == Algorithm.LINEAR:
            act_model = tf.keras.Sequential([
                # Take the last time-step.
                # Shape [batch, time, features] => [batch, 1, features]
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                # Shape => [batch, 1, out_steps*features]
                tf.keras.layers.Dense(self.setup.ACTUAL_SETUP.n_ahead * self.setup.ACTUAL_SETUP.num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([self.setup.ACTUAL_SETUP.n_ahead, self.setup.ACTUAL_SETUP.num_features])
            ])
        elif param.algorithm == Algorithm.DENSE:
            act_model = tf.keras.Sequential([
                # Take the last time step.
                # Shape [batch, time, features] => [batch, 1, features]
                tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                # Shape => [batch, 1, dense_units]
                tf.keras.layers.Dense(512, activation='relu'),
                # Shape => [batch, out_steps*features]
                tf.keras.layers.Dense(self.setup.ACTUAL_SETUP.n_ahead * self.setup.ACTUAL_SETUP.num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([self.setup.ACTUAL_SETUP.n_ahead, self.setup.ACTUAL_SETUP.num_features])
            ])
        elif param.algorithm == Algorithm.CONV:
            CONV_WIDTH = 3
            act_model = tf.keras.Sequential([
                # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                # Shape => [batch, 1, conv_units]
                tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
                # Shape => [batch, 1,  out_steps*features]
                tf.keras.layers.Dense(self.setup.ACTUAL_SETUP.n_ahead * self.setup.ACTUAL_SETUP.num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features]
                tf.keras.layers.Reshape([self.setup.ACTUAL_SETUP.n_ahead, self.setup.ACTUAL_SETUP.num_features])
            ])
        elif param.algorithm == Algorithm.LSTM:
            act_model = tf.keras.Sequential([
                # Shape [batch, time, features] => [batch, lstm_units].
                # Adding more `lstm_units` just overfits more quickly.
                tf.keras.layers.LSTM(32*param.lstm_units, return_sequences=False),
                # Shape => [batch, out_steps*features].
                tf.keras.layers.Dense(self.setup.ACTUAL_SETUP.n_ahead * self.setup.ACTUAL_SETUP.num_features,
                                      kernel_initializer=tf.initializers.zeros()),
                # Shape => [batch, out_steps, features].
                tf.keras.layers.Reshape([self.setup.ACTUAL_SETUP.n_ahead, self.setup.ACTUAL_SETUP.num_features])
            ])

        return act_model

    def compile_and_fit(self, model, window, params: ModelParameter):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=params.stop_loss,
                                                          patience=params.patience,
                                                          mode='min')

        """model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])"""

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=params.metrics)

        history = model.fit(window["train"], epochs=params.max_epochs,
                            validation_data=window["val"],
                            callbacks=[early_stopping])
        return history

    def save_model(self, model, name):
        model.save(self.setup.FILE_MODEL(name))

    def plot_history(self, hist):
        # summarize history for loss
        for key in hist.history:
            if not key.startswith("val"):
                plt.clf()
                plt.plot(hist.history[key])
                try:
                    plt.plot(hist.history["val_" + str(key)])
                except:
                    pass
                plt.title('model ' + key)
                plt.ylabel(key)
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(self.setup.FILE_MODEL_TRAIN(key))  # save as png

    def run_model(self, params, multi_window, id=-1):
        # create the model
        model = self.create_model(params)

        # fit the model and get the history
        history = self.compile_and_fit(model, multi_window, params)

        # save the model to use it later
        self.save_model(model, (str(id) if id > -1 else "all"))

        # plot the model train history
        self.plot_history(history)

    def start(self):
        # load the multi window

        # pre load the model setup
        params = self.setup.ACTUAL_SETUP.model_parameters

        if self.setup.ACTUAL_SETUP.id_handling == ID_HANDLING.SINGLE:
            for id in range(0, self.setup.ACTUAL_SETUP.pseudo_id_to_use):
                multi_window = self.load_sliding_window(id)
                self.run_model(params, multi_window, id)
        else:
            multi_window = self.load_sliding_window()
            self.run_model(params, multi_window)

