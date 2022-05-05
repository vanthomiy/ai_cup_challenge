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


def update_evaluation_file(mdl, perf, kys):
    # Store values in the csv file
    df = pd.read_csv(settings.FILE_EVALUATION_DATA)

    if settings.ACTUAL_SETUP.model_name in df["setup"].unique():
        index = df.loc[df['setup'] == settings.ACTUAL_SETUP.model_name].index[0]
        df.at[index, 'val'] = perf[kys[0]]
        df.at[index, 'val_loss'] = perf[kys[1]]
    else:
        df.loc[len(df.index)] = [settings.ACTUAL_SETUP.model_name, perf[kys[0]], perf[kys[1]]]

    df.to_csv(settings.FILE_EVALUATION_DATA, index=False)

    # Update the overview figure
    x = np.arange(len(df.index))

    width = 0.3

    metric_index = 1

    val_mae = [v[metric_index] for v in df["val"]]
    test_mae = [v[metric_index] for v in df["val_loss"]]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=df["setup"],
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    plt.savefig(settings.FILE_EVALUATION_OVERVIEW)


keys = ["val", "test"]

# load the model
model = load_model()

# load the data
data = load_sliding_window()

# evaluate the model
performance = evaluate_model(model, data, keys)

# store the values in a file and also update the overall picture
update_evaluation_file(model, performance, keys)
