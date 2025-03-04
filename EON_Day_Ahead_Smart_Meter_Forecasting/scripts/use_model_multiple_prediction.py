"""We use this to make multiple predictions between a timespan and store it in the right format for the
submission """

# we can do it for a list of dates indicating the first time to predict
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.setup import Normalization, ID_HANDLING, Timespan


class ModelMultiplePrediction:
    def __init__(self, setup):
        self.setup = setup

    def load_model(self, name):
        return tf.keras.models.load_model(self.setup.FILE_MODEL(name))

    def windowing(self, dfs, offset, id):
        wndw = {}
        if dfs is None:
            dfs = []

        for time in range(0, self.setup.ACTUAL_SETUP.time_windows_to_use):
            if len(dfs) <= time:
                df = pd.read_csv(self.setup.FILE_TIME_WINDOW_X(time))
                n = int((offset.days * 24) / (self.setup.ACTUAL_SETUP.data_interval.value / 2))
                if n != 0:
                    df1 = df.iloc[:-n]
                else:
                    df1 = df
                dfs.append(df1)
            if self.setup.ACTUAL_SETUP.id_handling == ID_HANDLING.SINGLE:
                pseudo_id = self.setup.PSEUDO_IDS[id]
                if pseudo_id not in wndw:
                    wndw[pseudo_id] = []

                features = [pseudo_id]
                features.extend(self.setup.ACTUAL_SETUP.features)
                features.extend(self.setup.ACTUAL_SETUP.weather_features)
                features.extend(["time"])
                df_id = dfs[time][features]
                df_id.rename(columns={pseudo_id: 'value'}, inplace=True)
                features = ["value"]
                features.extend(self.setup.ACTUAL_SETUP.features)
                features.extend(self.setup.ACTUAL_SETUP.weather_features)

                wndw[pseudo_id].append(df_id.tail(self.setup.ACTUAL_SETUP.n_before))
            else:
                for pseudo_id in self.setup.PSEUDO_IDS:
                    if pseudo_id not in wndw:
                        wndw[pseudo_id] = []

                    features = [pseudo_id]
                    features.extend(self.setup.ACTUAL_SETUP.features)
                    features.extend(self.setup.ACTUAL_SETUP.weather_features)
                    features.extend(["time"])
                    df_id = dfs[time][features]
                    df_id.rename(columns={pseudo_id: 'value'}, inplace=True)
                    features = ["value"]
                    features.extend(self.setup.ACTUAL_SETUP.features)
                    features.extend(self.setup.ACTUAL_SETUP.weather_features)

                    if self.setup.ACTUAL_SETUP.id_handling != ID_HANDLING.ALL:
                        df_id["pseudo_id"] = self.setup.PSEUDO_IDS.index(pseudo_id)

                    wndw[pseudo_id].append(df_id.tail(self.setup.ACTUAL_SETUP.n_before))

        return wndw, dfs

    def predict_values(self, mdl, wndw):
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
                    datetime_obj = datetime_obj + timedelta(hours=0,
                                                            minutes=self.setup.ACTUAL_SETUP.data_interval.value * 30)
                    datetime_string = str(datetime_obj)
                    result_for_id[datetime_string] = pred[0]

            results.append(result_for_id)

        return results

    def renormalize_data(self, df):
        _normalization = None

        with open(self.setup.FILE_NORMALIZATION_DATA, 'rb') as file:
            _normalization = pickle.load(file)

        ids = df.pop("pseudo_id")

        # load all counts
        counts = pd.read_csv(self.setup.FILE_COUNTS_DATA)
        if self.setup.ACTUAL_SETUP.normalization == Normalization.MEAN:
            df_un = (df * _normalization["std"] + _normalization["mean"])
        elif self.setup.ACTUAL_SETUP.normalization == Normalization.ZERO_TO_ONE:
            df_un = (df * _normalization["max"] + _normalization["min"])

        for pseudo_id in self.setup.PSEUDO_IDS:
            try:
                count_df = counts.loc[counts["pseudo_id"] == pseudo_id]
                count = count_df["n_dwellings"].iloc[0]
                df_un.iloc[self.setup.PSEUDO_IDS.index(pseudo_id)] *= count
            except Exception as ex:
                pass

        df_un.insert(0, 'pseudo_id', ids)
        return df_un

    def create_submission_format(self, dfs):
        preds = []

        for id in range(0, self.setup.ACTUAL_SETUP.pseudo_id_to_use):
            preds_for_id = {}
            for df in dfs:
                predicted_values = df.tail(int((7 * 24) / int(self.setup.ACTUAL_SETUP.data_interval.value / 2)))
                preds_for_id["pseudo_id"] = self.setup.PSEUDO_IDS[id]
                for index, row in predicted_values.iterrows():
                    preds_for_id[row["time"]] = row[self.setup.PSEUDO_IDS[id]]

            preds.append(preds_for_id)

        df = pd.DataFrame(preds)

        df.to_csv(self.setup.FILE_SUBMISSION_NORMED_DATA, index=False)

        df = self.renormalize_data(df)

        if self.setup.ACTUAL_SETUP.data_interval == Timespan.DAILY:
            columns = df.columns.values.tolist()[1:]
            for column in columns:
                df[column] = df[column] / 2

        df.to_csv(self.setup.FILE_SUBMISSION_DATA, index=False)

        return df
        # also save the un-normed values

        # now we have to map the time to the actual time...

    def create_submission_format_from_single(self, dfs):
        preds = []
        for id_index in range(0, len(dfs)):
            preds_for_id = {}
            for df in dfs[id_index]:
                predicted_values = df.tail(7 * 24 * int(self.setup.ACTUAL_SETUP.data_interval.value / 2))
                preds_for_id["pseudo_id"] = self.setup.PSEUDO_IDS[id_index]
                for index, row in predicted_values.iterrows():
                    preds_for_id[row["time"]] = row[self.setup.PSEUDO_IDS[id_index]]

            preds.append(preds_for_id)

        df = pd.DataFrame(preds)
        df.to_csv(self.setup.FILE_SUBMISSION_NORMED_DATA, index=False)

        df = self.renormalize_data(df)
        df.to_csv(self.setup.FILE_SUBMISSION_DATA, index=False)

        return df
        # also save the un-normed values

        # now we have to map the time to the actual time...

    def create_submission_daily(self, df_hourly):
        preds = []
        ids = df_hourly.pop("pseudo_id")

        for index, row in df_hourly.iterrows():
            preds_for_id = {}
            id_counts = {}

            for column in df_hourly.columns:
                date_obj = datetime.strptime(column, '%Y-%m-%d %H:%M:%S').date()

                if date_obj in preds_for_id:
                    preds_for_id[date_obj] += row[column]
                    id_counts[date_obj] += 1
                else:
                    preds_for_id[date_obj] = row[column]
                    id_counts[date_obj] = 1

            # for key in preds_for_id:
            #    preds_for_id[key] = preds_for_id[key] / id_counts[key]

            preds.append(preds_for_id)

        df = pd.DataFrame(preds)

        # df["pseudo_id"] = ids
        df.insert(0, 'pseudo_id', ids)

        df.to_csv(self.setup.FILE_SUBMISSION_DATA_DAILY, index=False)
        # also save the un-normed values

        # now we have to map the time to the actual time...

    def load_weather_data(self, offset):
        # startdate
        date = datetime(2017, 1, 1).date()
        days = timedelta(days=38) - (offset + timedelta(
            days=int(self.setup.ACTUAL_SETUP.n_before / (24 * self.setup.ACTUAL_SETUP.data_interval.value / 2))))

        # start_date = date + days
        start_date = datetime(2017, 1, 1).date()
        end_date = datetime(2019, 9, 10).date()

        # load the weather data
        weather_df = pd.read_csv(self.setup.FILE_WEATHER_DATA)

        """
        # make the dataset smaller. only columns and rows that are needed
        # exclude columns
        all_days = []
        for window in range(0, setup.ACTUAL_SETUP.time_windows_to_use - 1):
            days = [str(start_date + timedelta(days=days + window * 45)) for days in range(0, 7 + int(setup.ACTUAL_SETUP.n_before / (24 / setup.ACTUAL_SETUP.data_interval.value / 2)))]
            days = list(dict.fromkeys(days))
            all_days.extend(days)
        """

        # condition mask
        min_index = weather_df[weather_df['time'] == str(start_date)].index[0]
        max_index = weather_df[weather_df['time'] == str(end_date)].index[0]
        # mask = weather_df['time'].isin(all_days)
        # time = datetime.strptime(t, '%Y-%m-%d')

        # new dataframe with selected rows
        # df_wr = pd.DataFrame(weather_df[mask])
        df_wr = weather_df[min_index:max_index]
        time_df = df_wr.pop("time")

        # load normed data
        with open(self.setup.FILE_NORMALIZATION_DATA_WEATHER, 'rb') as file:
            normed = pickle.load(file)

        df_wr = (df_wr - normed["mean"]) / normed["std"]

        df_wr = df_wr.reset_index(drop=True)

        # df_wr["time"] = all_days
        df_wr["time"] = time_df.tolist()

        return df_wr

    def create_predictions(self, offset, a_id=-1):
        results = []

        model = self.load_model(str(a_id) if a_id > -1 else "all")
        if len(self.setup.ACTUAL_SETUP.weather_features) > 0:
            wather_data = self.load_weather_data(offset)

        dfs = None

        result = None
        # calculate how often we have to predict
        # iterations = int(
        #    (7 * 24) / (self.setup.ACTUAL_SETUP.n_ahead / (self.setup.ACTUAL_SETUP.data_interval.value / 2)))

        iterations = int(
            ((7 * 24) / self.setup.ACTUAL_SETUP.n_ahead) / (self.setup.ACTUAL_SETUP.data_interval.value / 2))

        for i in range(0, iterations):
            # create new window
            windows, dfs = self.windowing(dfs, offset, a_id)

            # predict values
            predictions = self.predict_values(model, windows)

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
            # counts = (self.setup.ACTUAL_SETUP.n_ahead / (self.setup.ACTUAL_SETUP.data_interval.value / 2))
            counts = (self.setup.ACTUAL_SETUP.n_ahead)  # / (self.setup.ACTUAL_SETUP.data_interval.value / 2))
            actual_count = 0
            for time in predicted_times:
                list_row = {}
                if a_id == -1:
                    for i in range(0, len(predicted_times[time])):
                        list_row[self.setup.PSEUDO_IDS[i]] = predicted_times[time][i]
                else:
                    list_row[self.setup.PSEUDO_IDS[a_id]] = predicted_times[time][0]

                list_row["time"] = time
                # Calculate seconds per day for the sin and cos functions with 24 hours/day * 60 minutes/hour * 60 seconds/minute
                seconds_per_day = 24 * 60 * 60
                # Calculate seconds per day for the sin and cos functions with seconds/day * days/year
                seconds_per_year = 365.2425 * seconds_per_day
                date_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                timestamp_s = int(round(date_time.timestamp()))

                _amplitude = 0.5
                _offset = 0.5

                if self.setup.ACTUAL_SETUP.normalization == Normalization.MEAN:
                    _amplitude = 2
                    _offset = 0

                list_row["day sin"] = _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_day)) + _offset
                list_row["day cos"] = _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_day)) + _offset
                list_row['year sin'] = _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_year)) + _offset
                list_row['year cos'] = _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_year)) + _offset

                if len(self.setup.ACTUAL_SETUP.weather_features) > 0:
                    try:
                        row = wather_data[wather_data["time"] == str(date_time.date())].iloc[0]
                    except Exception as ex:
                        print(ex)
                    for weather_features in self.setup.ACTUAL_SETUP.weather_features:
                        list_row[weather_features] = row[weather_features]
                print(actual_window)
                dfs[actual_window].loc[len(dfs[actual_window])] = list_row

                actual_count += 1
                if actual_count >= counts:
                    actual_window += 1
                    actual_count = 0

        return dfs

    def start(self):

        offset = timedelta(days=7)  # timedelta(days=0)
        dfs = None

        if self.setup.ACTUAL_SETUP.id_handling == ID_HANDLING.SINGLE:
            list_df = []
            for id in range(0, self.setup.ACTUAL_SETUP.pseudo_id_to_use):
                dfs_for_id = self.create_predictions(offset, id)
                list_df.append((dfs_for_id))
            sub_hourly = self.create_submission_format_from_single(list_df)
        else:
            dfs = self.create_predictions(offset)
            sub_hourly = self.create_submission_format(dfs)

        if self.setup.ACTUAL_SETUP.data_interval != Timespan.DAILY:
            self.create_submission_daily(sub_hourly)
