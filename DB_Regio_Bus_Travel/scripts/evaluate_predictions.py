"""This is used to evaluate the final predictions"""
import seaborn as sn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

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
        key_score = float(predicted_correct[key]["correct"]) / predicted_correct[key]["overall"]
        avg_score += key_score

    avg_score /= 4

    return avg_score


def find_data_for_prediction():
    """Takes the predicted data and searches and arranges the real data"""
    df_p = pd.read_csv(settings.FILE_SUBMISSION_DATA)
    df_r = pd.read_csv(settings.FILE_REGULAR_TRAVEL)

    y = []
    y_hat = []

    all_zones = df_r["EZone"].unique()

    df_p = df_p[df_p["EZone"].isin(all_zones)]

    df_p = df_p.head(1000)

    # get the times to be used
    all_times = df_p["date"].unique()
    df_r = df_r[df_r["date"].isin(all_times)]

    for index, row in df_p.iterrows():
        values = df_r[(df_r["date"] == row["date"]) & (df_r["EZone"] == row["EZone"]) & (df_r["hour"] == row["hour"])]
        if values.shape[0] <= 0:
            continue
        try:
            row_true = values.iloc[0]
            y.append(row["Passengers"])
            y_hat.append(row_true["Passengers"])
        except Exception as ex:
            # '5459 - Hauzenberg, Busbahnhof'
            print(row["EZone"])
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


def plot(df_p, df_r):

    df_p = [int(value) if value <= 3 else 3 for value in df_p]
    df_r = [int(value) if value <= 3 else 3 for value in df_r]

    array = confusion_matrix(df_r, df_p, normalize="true")

    DetaFrame_cm = pd.DataFrame(array)
    sn.heatmap(DetaFrame_cm, annot=True)
    plt.show()


df_p, df_r = find_data_for_prediction()

macro_f1_score = evaluate(df_p, df_r)

print(macro_f1_score)

update_evaluation_file(macro_f1_score)

plot(df_p, df_r)
