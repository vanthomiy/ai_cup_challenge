import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.applications.densenet import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, SpatialDropout1D, LSTM
from matplotlib import pyplot as plt

import settings as set
from scripts.setup import ModelParameter, Algorithm


class TrainModel:
    def __init__(self, setup):
        self.setup = setup

    def load_data(self):
        # load the pickle file with the windowed data
        df = pd.read_csv(self.setup.FILE_PRECESSED_DATA_TRAIN)

        X = df.iloc[:, 1:-2]
        Y = df.iloc[:, -1]

        # Convert target Y to one hot encoded Y for Neural Network
        Y = pd.get_dummies(Y)

        return X, Y

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def create_model(self, param, X, Y):
        act_model = None

        if param.algorithm == Algorithm.TRANSFORMER:
            act_model = tf.keras.Sequential()
            act_model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=128,
                        input_shape=[X.shape[0], X.shape[1]]
                    )
                )
            )
            act_model.add(tf.keras.layers.Dropout(rate=0.5))
            act_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
            act_model.add(tf.keras.layers.Dense(Y.shape[1], activation='softmax'))
        elif param.algorithm == Algorithm.LSTM:

            # Create model here
            act_model = Sequential()
            act_model.add(
                Dense(128, input_dim=X.shape[1], activation='relu'))  # Rectified Linear Unit Activation Function
            act_model.add(Dense(128, activation='relu'))
            act_model.add(Dense(8, activation='softmax'))  # Softmax for multi-class classification
            # Compile model here

        return act_model

    def get_callbacks(self):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=7,
                                       min_delta=0.0001)

        callbacks = [early_stopping]
        return callbacks

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def compile_and_fit(self, model, X, Y, params: ModelParameter):

        callbacks = self.get_callbacks()

        # we should take the same amount of classification data.
        # so we take n data for each at first
        val_split = 0.05
        count_per_class = 300
        dfs = []
        dfs_labels = []

        for i in range(0, 7):
            x1 = X.iloc[i * count_per_class:(i + 1) * count_per_class]
            y1 = Y.iloc[i * count_per_class:(i + 1) * count_per_class]
            # x1, y1 = self.unison_shuffled_copies(x1, y1)
            dfs.append(x1)
            dfs_labels.append(y1)

        X = None
        Y = None

        for i in range(0, len(dfs)):
            x = dfs[i][:int(count_per_class * (1 - val_split))]
            y = dfs_labels[i][:int(count_per_class * (1 - val_split))]
            if X is None:
                X = x
                Y = y
            else:
                X = X.append(x, ignore_index=True)
                Y = Y.append(y, ignore_index=True)

        for i in range(0, len(dfs)):
            x = dfs[i][int(count_per_class * (1 - val_split)):]
            y = dfs_labels[i][int(count_per_class * (1 - val_split)):]
            X = X.append(x, ignore_index=True)
            Y = Y.append(y, ignore_index=True)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        training = model.fit(X, Y, epochs=params.max_epochs, batch_size=16, validation_split=val_split,
                             callbacks=callbacks)

        accr = model.evaluate(X, Y)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        return training

    def save_model(self, model):
        model.save(self.setup.FILE_MODEL)

    def plot_history(self, hist):
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
                plt.savefig(self.setup.FILE_MODEL_TRAIN(key))  # save as png

    def start(self):
        # load the multi window
        X, Y = self.load_data()

        # pre load the model setup
        params = self.setup.ACTUAL_SETUP.model_parameters

        # create the model
        model = self.create_model(params, X, Y)

        # fit the model and get the history
        training = self.compile_and_fit(model, X, Y, params)

        # save the model to use it later
        self.save_model(model)

        # plot the model train history
        self.plot_history(training)
