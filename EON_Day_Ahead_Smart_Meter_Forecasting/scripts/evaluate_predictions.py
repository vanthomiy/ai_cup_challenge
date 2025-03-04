"""This is used to evaluate the final predictions"""
from datetime import datetime, time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.setup import Timespan


class EvaluatePredictions:
    def __init__(self, setup):
        self.setup = setup

    def evaluate(self, y, yhat, perc=True):
        """The evaluate function from the website readme"""
        y = y.drop('pseudo_id', axis=1).values
        yhat = yhat.drop('pseudo_id', axis=1).values
        n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)
        mape = None
        for i in range(n):
            error = []
            for a, f in zip(y[i], yhat[i]):
                # avoid division by 0
                if a > 0:
                    error.append(np.abs((a - f) / (a)))
            mape = np.mean(np.array(error))
        return mape * 100. if perc else mape

    def find_data_for_prediction(self):
        """Takes the predicted data and searches and arranges the real data"""
        df_p = pd.read_csv(self.setup.FILE_SUBMISSION_DATA)
        df_r = pd.read_csv(self.setup.FILE_TRAIN_DATA)
        # we have to adjust the headers first and then combine it to the daily values
        if self.setup.ACTUAL_SETUP.data_interval == Timespan.DAILY:
            # fake all headers here
            all_headers = []
            headers = df_p.columns.values.tolist()
            all_headers.append(headers[0])
            for header in headers[1:]:
                new_header = header
                for actual_time in range(0, 23):

                    time_string = str(actual_time)
                    if len(time_string) == 1:
                        time_string = "0" + time_string
                    new_header = header.replace("00:00:00", f"{time_string}:00:00")
                    all_headers.append(new_header)
            headers = all_headers
        else:
            headers = df_p.columns.values.tolist()

        df_rc = df_r[[*headers]]

        if self.setup.ACTUAL_SETUP.data_interval == Timespan.DAILY:
            # create daily from hourly
            df_rc = self.create_submission_daily(df_rc.copy())

        ids = df_p["pseudo_id"].tolist()
        df_rcr = df_rc[df_rc['pseudo_id'].isin(ids)]
        return df_p, df_rcr

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

            preds.append(preds_for_id)

        df = pd.DataFrame(preds)

        # df["pseudo_id"] = ids
        df.insert(0, 'pseudo_id', ids)

        return df
        # also save the un-normed values

        # now we have to map the time to the actual time...

    def find_data_for_prediction_daily(self, df_r):
        """Takes the predicted data and searches and arranges the real data"""
        df_p_daily = pd.read_csv(self.setup.FILE_SUBMISSION_DATA_DAILY)
        df_rcr_daily = self.create_submission_daily(df_r.copy())
        return df_p_daily, df_rcr_daily

    def update_evaluation_file(self, value_hourly, value_daily):
        # Store values in the csv file
        df = pd.read_csv(self.setup.FILE_MAPE_EVALUATION_DATA)

        if self.setup.SETUP_KEY in df["setup"].unique():
            index = df.loc[df['setup'] == self.setup.SETUP_KEY].index[0]
            df.at[index, 'value_hourly'] = value_hourly
            df.at[index, 'value_daily'] = value_daily
        else:
            df.loc[len(df.index)] = [self.setup.SETUP_KEY, value_hourly, value_daily]

        df.to_csv(self.setup.FILE_MAPE_EVALUATION_DATA, index=False)

        # Update the overview figure
        x = np.arange(len(df.index))

        width = 0.3

        metric_index = 1
        val_mae = df["value_daily"].tolist()

        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.xticks(ticks=x, labels=df["setup"].tolist(), rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        plt.savefig(self.setup.FILE_MAPE_EVALUATION_OVERVIEW)

    def plot(self, df_p, df_r, counts=24):
        plt.clf()

        row = df_p.iloc[0]
        a = row.T
        data = a[::counts].tolist()[1:]
        plt.plot(data, color='magenta', marker='o', mfc='pink')  # plot the data
        plt.xticks(range(0, len(data) + 1, 1))  # set the tick frequency on x-axis

        row = df_r.iloc[0]
        a = row.T
        data = a[::counts].tolist()[1:]
        plt.plot(data, color='red', marker='x', mfc='pink')  # plot the data
        plt.xticks(range(0, len(data) + 1, 1))  # set the tick frequency on x-axis

        plt.ylabel('data')  # set the label for y axis
        plt.xlabel('index')  # set the label for x-axis
        plt.title("Plotting a list")  # set the title of the graph
        # plt.show()  # display the graph
        plt.savefig(self.setup.FILE_MAPE_EVALUATION_TIMESERIES)

    def start(self):
        df_p, df_r = self.find_data_for_prediction()

        if self.setup.ACTUAL_SETUP.data_interval == Timespan.HOURLY:
            mape = self.evaluate(df_p, df_r)
            df_p_daily, df_r_daily = self.find_data_for_prediction_daily(df_r)
            mape_daily = self.evaluate(df_p_daily, df_r_daily)
            print(mape)
            print(mape_daily)
            print((mape + mape_daily) / 2)
            self.update_evaluation_file(mape, mape_daily)
            self.plot(df_p_daily, df_r_daily, 1)
        else:
            #df_r_d = self.create_submission_daily(df_r.copy())
            columns = df_p.columns.values.tolist()[1:]
            #for column in columns:
            #    df_p[column] = df_p[column] / 2

            mape = self.evaluate(df_r, df_p)
            print(mape)
            self.update_evaluation_file(mape, mape)

        self.plot(df_p, df_r)

