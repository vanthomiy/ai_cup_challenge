"""This is used to evaluate the final predictions"""
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import settings


def evaluate(y, yhat, perc=True):
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


def find_data_for_prediction():
    """Takes the predicted data and searches and arranges the real data"""
    df_p = pd.read_csv(settings.FILE_SUBMISSION_DATA)
    df_r = pd.read_csv(settings.FILE_TRAIN_DATA)
    headers = df_p.columns.values.tolist()
    df_rc = df_r[[*headers]]
    ids = df_p["pseudo_id"].tolist()
    df_rcr = df_rc[df_rc['pseudo_id'].isin(ids)]
    return df_p, df_rcr


def create_submission_daily(df_hourly):
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


def find_data_for_prediction_daily(df_r):
    """Takes the predicted data and searches and arranges the real data"""
    df_p_daily = pd.read_csv(settings.FILE_SUBMISSION_DATA_DAILY)
    df_rcr_daily = create_submission_daily(df_r.copy())
    return df_p_daily, df_rcr_daily


def update_evaluation_file(value):
    # Store values in the csv file
    df = pd.read_csv(settings.FILE_MAPE_EVALUATION_DATA)

    if settings.ACTUAL_SETUP.model_name in df["setup"].unique():
        index = df.loc[df['setup'] == settings.ACTUAL_SETUP.model_name].index[0]
        df.at[index, 'value'] = value
    else:
        df.loc[len(df.index)] = [settings.ACTUAL_SETUP.model_name, value]

    df.to_csv(settings.FILE_MAPE_EVALUATION_DATA, index=False)

    # Update the overview figure
    x = np.arange(len(df.index))

    width = 0.3

    metric_index = 1
    val_mae = df["value"].tolist()

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.xticks(ticks=x, labels=df["setup"].tolist(), rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    plt.savefig(settings.FILE_MAPE_EVALUATION_OVERVIEW)


def plot(df_p, df_r):
    plt.clf()

    row = df_p.iloc[0]
    a = row.T
    data = a[::24].tolist()[1:]
    plt.plot(data, color='magenta', marker='o', mfc='pink')  # plot the data
    plt.xticks(range(0, len(data) + 1, 1))  # set the tick frequency on x-axis

    row = df_r.iloc[0]
    a = row.T
    data = a[::24].tolist()[1:]
    plt.plot(data, color='red', marker='x', mfc='pink')  # plot the data
    plt.xticks(range(0, len(data) + 1, 1))  # set the tick frequency on x-axis

    plt.ylabel('data')  # set the label for y axis
    plt.xlabel('index')  # set the label for x-axis
    plt.title("Plotting a list")  # set the title of the graph
    plt.show()  # display the graph
    plt.savefig(settings.FILE_MAPE_EVALUATION_TIMESERIES)


df_p, df_r = find_data_for_prediction()
df_p_daily, df_r_daily = find_data_for_prediction_daily(df_r)

mape = evaluate(df_p, df_r)
mape_daily = evaluate(df_p_daily, df_r_daily)

print(mape)
print(mape_daily)

print((mape + mape_daily) / 2)

update_evaluation_file(mape)

plot(df_p, df_r)
plot(df_p_daily, df_r_daily)
