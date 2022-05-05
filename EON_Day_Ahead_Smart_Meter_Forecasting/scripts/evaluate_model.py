import pickle

import tensorflow as tf

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


keys = ["val", "test"]

# load the model
model = load_model()

# load the data
data = load_data()

# evaluate the model
performance = evaluate_model(model, data)

# store the values in a file


# update the overall picture

