"""We use this to make a single prediction starting from a specific date and store it in the right format for the
submission """

# we can do it for a list of dates indicating the first time to predict
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

import settings


def load_model():
    return tf.keras.models.load_model(settings.FILE_MODEL)


def windowing():
    wndw = {}

    for time in range(0, settings.ACTUAL_SETUP.time_windows_to_use):
        df = pd.read_csv(settings.FILE_TIME_WINDOW_X(time))
        for pseudo_id in settings.PSEUDO_IDS:
            if pseudo_id not in wndw:
                wndw[pseudo_id] = []

            features = [pseudo_id]
            features.extend(settings.ACTUAL_SETUP.features)
            features.extend(["time"])
            df_id = df[features]
            df_id.rename(columns={pseudo_id: 'value'}, inplace=True)
            features = ["value"]
            features.extend(settings.ACTUAL_SETUP.features)
            df_id["pseudo_id"] = settings.PSEUDO_IDS.index(pseudo_id)

            wndw[pseudo_id].append(df_id.tail(settings.ACTUAL_SETUP.n_before))

    return wndw


def predict_values(mdl, wndw):
    results = []
    for pseudo_id in wndw:
        result_for_id = {"pseudo_id": pseudo_id}
        for time in wndw[pseudo_id]:
            # we have to get the actual date of the id...
            # start time
            actual_time_string = time["time"].tolist()[-1]
            datetime_obj = datetime.strptime(actual_time_string, '%Y-%m-%d %H:%M:%S')
            time.pop("time")

            arr_data = np.array(time)

            all_inputs = np.array([arr_data])

            predictions = mdl.predict(all_inputs)

            for pred in predictions[0]:
                datetime_obj = datetime_obj + timedelta(hours=0, minutes=settings.ACTUAL_SETUP.data_interval.value * 30)
                datetime_string = str(datetime_obj)
                result_for_id[datetime_string] = pred[0]

        results.append(result_for_id)

    return results


def renormalize_data(df):
    _normalization = None

    with open(settings.FILE_NORMALIZATION_DATA, 'rb') as file:
        _normalization = pickle.load(file)

    ids = df.pop("pseudo_id")

    # load all counts
    counts = pd.read_csv(settings.FILE_COUNTS_DATA)

    df_un = (df * _normalization["std"] + _normalization["mean"])

    for pseudo_id in settings.PSEUDO_IDS:
        count = counts.loc[counts.pseudo_id == pseudo_id]["n_dwellings"][0]
        df_un.iloc[settings.PSEUDO_IDS.index(pseudo_id)] *= count

    df_un.insert(0, 'pseudo_id', ids)
    return df_un


def create_submission_format(preds):
    df = pd.DataFrame(preds)
    df.to_csv(settings.FILE_SUBMISSION_NORMED_DATA, index=False)

    df = renormalize_data(df)
    df.to_csv(settings.FILE_SUBMISSION_DATA, index=False)

    # also save the un-normed values

    # now we have to map the time to the actual time...


# create windows and predictions for each time

results = []

model = load_model()

windows = windowing()

predictions = predict_values(model, windows)

create_submission_format(predictions)
