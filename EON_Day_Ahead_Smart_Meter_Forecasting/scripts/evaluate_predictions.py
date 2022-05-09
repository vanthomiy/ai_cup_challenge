"""This is used to evaluate the final predictions"""
import numpy as np
import pandas as pd


def evaluate(y, yhat, perc=True):
    y = y.drop('pseudo_id', axis = 1).values
    yhat = yhat.drop('pseudo_id', axis = 1).values
    n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)
    for i in range(n):
        error = []
        for a, f in zip(y[i], yhat[i]):
            # avoid division by 0
            if a > 0:
                error.append(np.abs((a - f)/(a)))
        mape = np.mean(np.array(error))
    return mape * 100. if perc else mape


def find_data_for_prediction():
