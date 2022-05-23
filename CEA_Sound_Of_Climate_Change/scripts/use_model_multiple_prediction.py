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


class ModelMultiplePrediction:
    def __init__(self, setup):
        self.setup = setup

    def load_model(self):
        return tf.keras.models.load_model(self.setup.FILE_MODEL)

    def load_data(self):
        # load the pickle file with the windowed data
        df = pd.read_csv(self.setup.FILE_PRECESSED_DATA_TEST)
        df = df.iloc[:, 1:-1]

        X = df.values

        return X

    def predict_values(self, mdl, data):

        predictions = mdl.predict(data)

        return predictions

    def create_submission_format(self, preds):
        preds = [pred.tolist().index(max(pred)) for pred in preds]

        df = pd.DataFrame(preds, columns=["PredictedClass"])

        df.to_csv(self.setup.FILE_SUBMISSION_DATA, index=False)

        return df
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

    def start(self):
        results = []

        model = self.load_model()

        # create new window
        dfs = self.load_data()

        # predict values
        predictions = self.predict_values(model, dfs)

        df = self.create_submission_format(predictions)

        df.to_csv(self.setup.FILE_SUBMISSION_DATA, index=False)
