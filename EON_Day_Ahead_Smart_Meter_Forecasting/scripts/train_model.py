import pickle

import tensorflow as tf
from matplotlib import pyplot as plt

import settings
from scripts.setup import ModelParameter, Algorithm


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
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=params.patience,
                                                      mode='min')

    model.compile(loss=params.loss,
                  optimizer=params.optimizer,
                  metrics=params.metrics)

    history = model.fit(window.train, epochs=params.max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def save_model(model):
    model.save(settings.FILE_MODEL)


def plot_history(hist):
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(settings.FILE_MODEL_TRAIN) #save as png


# load the pickle file with the windowed data
with open(settings.FILE_WINDOWED_DATA, 'rb') as f:
    multi_window = pickle.load(f)

if multi_window is None:
    print("No multi_window was found")

params = settings.ACTUAL_SETUP.model_parameters

# create the model
model = create_model(params)

# fit the model and get the history
history = compile_and_fit(model, multi_window, params)

# save the model to use it later
save_model(model)

# plot the model train history
plot_history(history)
