import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

import settings


def load_model():
    return tf.keras.models.load_model(settings.FILE_MODEL)


def load_data():
    windowed_data = {}
    with open(settings.FILE_WINDOWED_DATA, 'rb') as f:
        windowed_data = pickle.load(f)

    return windowed_data


def evaluate_model(mdl, dat, kys):
    perf = {}

    for key in kys:
        perf[key] = mdl.evaluate(dat[key])

    return perf


def update_evaluation_file(mdl, perf, kys):
    # Store values in the csv file
    df = pd.read_csv(settings.FILE_EVALUATION_DATA)

    if settings.ACTUAL_SETUP.model_name in df["setup"].unique():
        df["setup" == settings.ACTUAL_SETUP.model_name].iloc[0] = [settings.ACTUAL_SETUP.model_name, perf[kys[0]], perf[kys[1]]]
    else:
        df.loc[len(df.index)] = [settings.ACTUAL_SETUP.model_name, perf[kys[0]], perf[kys[1]]]

    df.to_csv(settings.FILE_EVALUATION_DATA)

    # Update the overview figure
    x = np.arange(len(df.index))

    width = 0.3

    metric_name = 'mean_absolute_error'
    metric_index = mdl.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in df[kys[0]].values()]
    test_mae = [v[metric_index] for v in df[kys[1]].values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=df["setup"].values(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    plt.savefig(settings.FILE_EVALUATION_OVERVIEW)


keys = ["val", "test"]

# load the model
model = load_model()

# load the data
data = load_data()

# evaluate the model
performance = evaluate_model(model, data, keys)

# store the values in a file and also update the overall picture
update_evaluation_file(model, performance, keys)
