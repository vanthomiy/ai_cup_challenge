import pickle

import tensorflow as tf
from matplotlib import pyplot as plt

import settings
from scripts.setup import ModelParameter, Algorithm


def load_sliding_window():
    # load the pickle file with the windowed data
    mv_ds = {}

    for value in settings.TEST_TRAIN_VALID:
        ds = tf.data.experimental.load(
            settings.FILE_WINDOWED_DATA(value))
        mv_ds[value] = ds

    return mv_ds


def create_model(param):
    act_model = None

    if param.algorithm == Algorithm.LINEAR:
        act_model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(settings.ACTUAL_SETUP.n_ahead * settings.ACTUAL_SETUP.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([settings.ACTUAL_SETUP.n_ahead, settings.ACTUAL_SETUP.num_features])
        ])
    elif param.algorithm == Algorithm.DENSE:
        act_model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(settings.ACTUAL_SETUP.n_ahead * settings.ACTUAL_SETUP.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([settings.ACTUAL_SETUP.n_ahead, settings.ACTUAL_SETUP.num_features])
        ])
    elif param.algorithm == Algorithm.CONV:
        CONV_WIDTH = 3
        act_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(settings.ACTUAL_SETUP.n_ahead * settings.ACTUAL_SETUP.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([settings.ACTUAL_SETUP.n_ahead, settings.ACTUAL_SETUP.num_features])
        ])
    elif param.algorithm == Algorithm.LSTM:
        act_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(settings.ACTUAL_SETUP.n_ahead * settings.ACTUAL_SETUP.num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([settings.ACTUAL_SETUP.n_ahead, settings.ACTUAL_SETUP.num_features])
        ])

    return act_model


def compile_and_fit(model, window, params: ModelParameter):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="mean_absolute_percentage_error", # 'mean_absolute_error',
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


def save_model(model):
    model.save(settings.FILE_MODEL)


def plot_history(hist):
    # summarize history for loss
    for key in hist.history:
        if not key.startswith("val"):
            plt.clf()
            plt.plot(hist.history[key])
            plt.plot(hist.history["val_" + str(key)])
            plt.title('model ' + key)
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(settings.FILE_MODEL_TRAIN(key))  # save as png





# load the multi window
multi_window = load_sliding_window()

# pre load the model settings
params = settings.ACTUAL_SETUP.model_parameters

# create the model
model = create_model(params)

# fit the model and get the history
history = compile_and_fit(model, multi_window, params)

# save the model to use it later
save_model(model)

# plot the model train history
plot_history(history)
