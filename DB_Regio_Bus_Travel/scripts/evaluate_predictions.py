"""This is used to evaluate the final predictions"""
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import settings


def evaluate(y, yhat, perc=True):
    prediction_on_real_scores = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    predicted_correct = {
        0: {"correct": 0, "overall": 0},
        1: {"correct": 0, "overall": 0},
        2: {"correct": 0, "overall": 0},
        3: {"correct": 0, "overall": 0}
    }

    for i in range(0, len(yhat)):
        count_real = int(yhat[i]) if int(yhat[i]) <= 3 else 3
        count_predicted = int(y[i]) if int(y[i]) <= 3 else 3

        prediction_on_real_scores[count_real].append(count_predicted)
        predicted_correct[count_real]["overall"] += 1
        if count_real == count_predicted:
            predicted_correct[count_real]["correct"] += 1

    avg_score = 0

    for key in predicted_correct:
        key_score = predicted_correct[key]["correct"] / predicted_correct[key]["overall"]
        avg_score += key_score

    avg_score /= 4

    return avg_score


def find_data_for_prediction():
    """Takes the predicted data and searches and arranges the real data"""
    df_p = pd.read_csv(settings.FILE_SUBMISSION_DATA)
    df_r = pd.read_csv(settings.FILE_REGULAR_TRAVEL)

    y = []
    y_hat = []

    for index, row in df_p.iterrows():
        values = df_r[(df_r["date"] == row["date"]) & (df_r["EZone"] == row["EZone"]) & (df_r["hour"] == row["hour"])]
        try:
            row_true = values.iloc[0]
            y.append(row["Passengers"])
            y_hat.append(row_true["Passengers"])
        except:
            '5459 - Hauzenberg, Busbahnhof'
            pass

    return y, y_hat


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


def plot(df_p, df_r, counts=24):
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
    plt.show()  # display the graph
    plt.savefig(settings.FILE_MAPE_EVALUATION_TIMESERIES)


df_p, df_r = find_data_for_prediction()

macro_f1_score = evaluate(df_p, df_r)

print(macro_f1_score)

update_evaluation_file(macro_f1_score)
