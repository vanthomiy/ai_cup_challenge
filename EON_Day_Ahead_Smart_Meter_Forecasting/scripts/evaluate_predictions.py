"""This is used to evaluate the final predictions"""
import numpy as np
import pandas as pd

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


df_p, df_r = find_data_for_prediction()
mape = evaluate(df_p, df_r)
print(mape)
