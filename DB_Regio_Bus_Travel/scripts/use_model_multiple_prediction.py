"""We use this to make multiple predictions between a timespan and store it in the right format for the
submission """

# we can do it for a list of dates indicating the first time to predict
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

import settings
from scripts.setup import Normalization


def load_model():
    return tf.keras.models.load_model(settings.FILE_MODEL)


def windowing(dfs):
    wndw = {}
    if dfs is None:
        dfs = []

    for time in range(0, settings.ACTUAL_SETUP.time_windows_to_use):
        if len(dfs) <= time:
            df = pd.read_csv(settings.FILE_TIME_WINDOW_X(time))
            n = int(offset.days * 24 * settings.ACTUAL_SETUP.data_interval.value)
            if n != 0:
                df1 = df.iloc[:-n]
            else:
                df1 = df
            dfs.append(df1)
        for pseudo_id_int in settings.BUS_STOPS:
            pseudo_id = str(pseudo_id_int)
            if dfs[time][pseudo_id].isnull().values.any():
                continue

            if pseudo_id not in wndw:
                wndw[pseudo_id] = []

            features = [pseudo_id]
            features.extend(settings.ACTUAL_SETUP.features)
            features.extend(settings.ACTUAL_SETUP.weather_features)
            features.extend(["time"])
            df_id = dfs[time][features]
            df_id.rename(columns={pseudo_id: 'value'}, inplace=True)
            features = ["value"]
            features.extend(settings.ACTUAL_SETUP.features)
            features.extend(settings.ACTUAL_SETUP.weather_features)
            id = settings.BUS_STOPS.index(pseudo_id_int)
            df_id["pseudo_id"] = id

            wndw[pseudo_id].append(df_id.tail(settings.ACTUAL_SETUP.n_before))

    return wndw, dfs


def predict_values(mdl, wndw):
    results = []
    for pseudo_id in wndw:
        result_for_id = {"pseudo_id": pseudo_id}
        for time in wndw[pseudo_id]:
            # we have to get the actual date of the id...
            # start time
            actual_time_string = time["time"].tolist()[-1]
            datetime_obj = datetime.strptime(actual_time_string, '%Y-%m-%d %H:%M:%S')
            datetime_obj = datetime_obj  # - offset
            time.pop("time")

            arr_data = np.array(time)

            all_inputs = np.array([arr_data])

            predictions = mdl.predict(all_inputs)

            for pred in predictions[0]:
                datetime_obj = datetime_obj + timedelta(hours=0, minutes=settings.ACTUAL_SETUP.data_interval.value * 60)
                datetime_string = str(datetime_obj)
                result_for_id[datetime_string] = pred[0]

        results.append(result_for_id)

    return results


def renormalize_data(df):
    _normalization = None

    with open(settings.FILE_NORMALIZATION_DATA, 'rb') as file:
        _normalization = pickle.load(file)

    ids = df.pop("Passengers")

    # load all counts
    # counts = pd.read_csv(settings.FILE_COUNTS_DATA)

    ids_un = (ids * _normalization["std"] + _normalization["mean"])

    ids_un = [int(value) if value <= 3 else 3 for value in ids_un]

    df["Passengers"] = ids_un # .tolist()
    return df


def create_submission_format(dfs):
    preds = []
    for df in dfs:
        predicted_values = df.tail(7 * 24 * settings.ACTUAL_SETUP.data_interval.value)
        for day in range(0, 7):
            predicted_values_days = predicted_values[day*24:(day+1) * 24]
            for bus_id in settings.BUS_STOPS_SORTED:
                if bus_id not in predicted_values_days.columns or predicted_values_days[bus_id].isnull().values.any():
                    continue

                for index, row in predicted_values_days.iterrows():
                    datetime_obj = datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S')

                    name = f"{str(bus_id)} - {settings.BUS_STOPS_DICT[str(bus_id)]}"

                    prediction = {
                        "date": datetime_obj.date(),
                        "EZone": name,
                        "hour": datetime_obj.hour,
                        "Passengers": row[bus_id]
                    }
                    preds.append(prediction)

            """for id in range(0, settings.ACTUAL_SETUP.bus_stops_to_us):
                if id >= len(settings.BUS_STOPS):
                    continue
                bus_id = str(settings.BUS_STOPS[id])
                if bus_id not in predicted_values_days.columns or predicted_values_days[bus_id].isnull().values.any():
                    continue

                for index, row in predicted_values_days.iterrows():
                    datetime_obj = datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S')

                    name = f"{str(bus_id)} - {settings.BUS_STOPS_DICT[str(bus_id)]}"

                    prediction = {
                        "date": datetime_obj.date(),
                        "EZone": name,
                        "hour": datetime_obj.hour,
                        "Passengers": row[bus_id]
                    }
                    preds.append(prediction)
"""
    df = pd.DataFrame(preds)
    df.to_csv(settings.FILE_SUBMISSION_NORMED_DATA, index=False)

    df = renormalize_data(df)
    df.to_csv(settings.FILE_SUBMISSION_DATA, index=False)

    return df
    # also save the un-normed values

    # now we have to map the time to the actual time...



def load_weather_data(offset):
    # startdate
    date = datetime(2017, 1, 1).date()
    days = timedelta(days=38) - offset

    start_date = date + days

    # load the weather data
    weather_df = pd.read_csv(settings.FILE_WEATHER_DATA)

    # make the dataset smaller. only columns and rows that are needed
    # exclude columns
    all_days = []
    for window in range(0, settings.ACTUAL_SETUP.time_windows_to_use):
        days = [str(start_date + timedelta(days=days + window * 45)) for days in range(0, 7)]
        days = list(dict.fromkeys(days))
        all_days.extend(days)

    # condition mask
    mask = weather_df['time'].isin(all_days)

    # new dataframe with selected rows
    df_wr = pd.DataFrame(weather_df[mask])

    # load normed data
    with open(settings.FILE_NORMALIZATION_DATA_WEATHER, 'rb') as file:
        normed = pickle.load(file)

    df_wr = (df_wr - normed["mean"]) / normed["std"]

    df_wr = df_wr.reset_index(drop=True)

    df_wr["time"] = all_days

    return df_wr


# create windows and predictions for each time

offset = timedelta(days=0)  # timedelta(days=0)

results = []

model = load_model()

# wather_data = load_weather_data(offset)

dfs = None

result = None
# calculate how often we have to predict
iterations = int((7 * 24) / (settings.ACTUAL_SETUP.n_ahead / settings.ACTUAL_SETUP.data_interval.value))

for i in range(0, iterations):
    print("predict " + str(i))
    # create new window
    windows, dfs = windowing(dfs)

    # predict values
    predictions = predict_values(model, windows)

    predicted_times = {}

    # add predictions to the actual dataframe
    for id in predictions:
        for time in id:
            if time == 'pseudo_id':
                continue

            if time not in predicted_times:
                predicted_times[time] = []

            predicted_times[time].append(id[time])

    actual_window = 0
    counts = (settings.ACTUAL_SETUP.n_ahead / settings.ACTUAL_SETUP.data_interval.value)
    actual_count = 0
    for time in predicted_times:
        list_row = {}
        for i in range(0, len(predicted_times[time])):
            list_row[settings.BUS_STOPS[i]] = predicted_times[time][i]

        list_row["time"] = time
        # Calculate seconds per day for the sin and cos functions with 24 hours/day * 60 minutes/hour * 60 seconds/minute
        seconds_per_day = 24 * 60 * 60
        # Calculate seconds per day for the sin and cos functions with seconds/day * days/year
        seconds_per_year = 365.2425 * seconds_per_day
        date_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        timestamp_s = int(round(date_time.timestamp()))

        _amplitude = 0.5
        _offset = 0.5

        if settings.ACTUAL_SETUP.normalization == Normalization.MEAN:
            _amplitude = 2
            _offset = 0

        list_row["day sin"] = _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_day)) + _offset
        list_row["day cos"] = _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_day)) + _offset
        list_row['year sin'] = _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_year)) + _offset
        list_row['year cos'] = _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_year)) + _offset

        # row = wather_data[wather_data["time"] == str(date_time.date())].iloc[0]
        # for weather_features in settings.ACTUAL_SETUP.weather_features:
        #    list_row[weather_features] = row[weather_features]

        dfs[actual_window].loc[len(dfs[actual_window])] = list_row

        actual_count += 1
        if actual_count >= counts:
            actual_window += 1
            actual_count = 0

sub_hourly = create_submission_format(dfs)

