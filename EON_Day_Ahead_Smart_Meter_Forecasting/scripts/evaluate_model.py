import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from scripts.setup import ID_HANDLING


class EvaluateModel:
    def __init__(self, setup):
        self.setup = setup

    def load_model(self, name):
        return tf.keras.models.load_model(self.setup.FILE_MODEL(name))

    def load_sliding_window(self, id=-1):
        # load the pickle file with the windowed data
        mv_ds = {}

        for value in self.setup.TEST_TRAIN_VALID:
            ds = tf.data.experimental.load(
                self.setup.FILE_WINDOWED_DATA(value, (str(id) if id > -1 else "all")))
            mv_ds[value] = ds

        return mv_ds

    def evaluate_model(self, mdl, dat, kys):
        perf = {}

        for key in kys:
            perf[key] = mdl.evaluate(dat[key])

        return perf

    def update_evaluation_file(self, perf, kys):
        # Store values in the csv file
        df = pd.read_csv(self.setup.FILE_EVALUATION_DATA)

        if self.setup.SETUP_KEY in df["setup"].unique():
            index = df.loc[df['setup'] == self.setup.SETUP_KEY].index[0]
            df.at[index, 'val'] = perf[kys[0]][0]
            df.at[index, 'val_loss'] = perf[kys[0]][1]
            df.at[index, 'test'] = perf[kys[1]][0]
            df.at[index, 'test_loss'] = perf[kys[1]][1]
        else:
            df.loc[len(df.index)] = [self.setup.SETUP_KEY,
                                     perf[kys[0]][0],
                                     perf[kys[0]][1],
                                     perf[kys[1]][0],
                                     perf[kys[1]][1]]

        df.to_csv(self.setup.FILE_EVALUATION_DATA, index=False)

        # Update the overview figure
        x = np.arange(len(df.index))

        width = 0.3

        metric_index = 1
        val_mae = df["val"].tolist()
        test_mae = df["test"].tolist()

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=df["setup"].tolist(), rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        plt.savefig(self.setup.FILE_EVALUATION_OVERVIEW)

    def plot(self, data, model=None, plot_col='value', max_subplots=3):
        total_window_size = self.setup.ACTUAL_SETUP.n_ahead + self.setup.ACTUAL_SETUP.n_before

        input_slice = slice(0, self.setup.ACTUAL_SETUP.n_before)
        input_indices = np.arange(total_window_size)[input_slice]

        label_start = total_window_size - self.setup.ACTUAL_SETUP.n_ahead
        labels_slice = slice(label_start, None)
        label_indices = np.arange(total_window_size)[labels_slice]

        inputs, labels = next(iter(data))
        plt.figure(figsize=(12, 8))
        plot_col_index = 0  # column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel(f"Time steps: {self.setup.ACTUAL_SETUP.data_interval}")
        plt.savefig(self.setup.FILE_EVALUATION_TIMESERIES)

    def evaluate(self, keys, id=-1):

        # load the model
        model = self.load_model(str(id) if id > -1 else "all")

        # load the data
        data = self.load_sliding_window(id)

        # evaluate the model
        return self.evaluate_model(model, data, keys), data, model

    def start(self):
        keys = ["val", "test"]

        if self.setup.ACTUAL_SETUP.id_handling == ID_HANDLING.SINGLE:
            for id in range(0, self.setup.ACTUAL_SETUP.pseudo_id_to_use):
                self.evaluate(keys, id)
        else:
            performance, data, model = self.evaluate(keys)
            # store the values in a file and also update the overall picture
            self.update_evaluation_file(performance, keys)

            # create predictions and store them as pictures
            self.plot(data=data["test"], model=model)
