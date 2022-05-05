import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

import settings


def load_model():
    return tf.keras.models.load_model(settings.FILE_MODEL)


def load_sliding_window():
    # load the pickle file with the windowed data
    mv_ds = {}

    for value in settings.TEST_TRAIN_VALID:
        ds = tf.data.experimental.load(
            settings.FILE_WINDOWED_DATA(value))
        mv_ds[value] = ds

    return mv_ds


def evaluate_model(mdl, dat, kys):
    perf = {}

    for key in kys:
        perf[key] = mdl.evaluate(dat[key])

    return perf


def update_evaluation_file(perf, kys):
    # Store values in the csv file
    df = pd.read_csv(settings.FILE_EVALUATION_DATA)

    if settings.ACTUAL_SETUP.model_name in df["setup"].unique():
        index = df.loc[df['setup'] == settings.ACTUAL_SETUP.model_name].index[0]
        df.at[index, 'val'] = perf[kys[0]][0]
        df.at[index, 'val_loss'] = perf[kys[0]][1]
        df.at[index, 'test'] = perf[kys[1]][0]
        df.at[index, 'test_loss'] = perf[kys[1]][1]
    else:
        df.loc[len(df.index)] = [settings.ACTUAL_SETUP.model_name,
                                 perf[kys[0]][0],
                                 perf[kys[0]][1],
                                 perf[kys[1]][0],
                                 perf[kys[1]][1]]

    df.to_csv(settings.FILE_EVALUATION_DATA, index=False)

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
    plt.savefig(settings.FILE_EVALUATION_OVERVIEW)


def plot(data, model=None, plot_col='value', max_subplots=3):
    total_window_size = settings.ACTUAL_SETUP.n_ahead + settings.ACTUAL_SETUP.n_before

    input_slice = slice(0, settings.ACTUAL_SETUP.n_ahead)
    input_indices = np.arange(total_window_size)[input_slice]

    label_start = total_window_size - settings.ACTUAL_SETUP.n_ahead
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

    plt.xlabel('Time [h]')
    plt.savefig(settings.FILE_EVALUATION_TIMESERIES)


keys = ["val", "test"]

# load the model
model = load_model()

# load the data
data = load_sliding_window()

# evaluate the model
performance = evaluate_model(model, data, keys)

# store the values in a file and also update the overall picture
update_evaluation_file(performance, keys)

# create predictions and store them as pictures
plot(data=data["test"], model=model)
