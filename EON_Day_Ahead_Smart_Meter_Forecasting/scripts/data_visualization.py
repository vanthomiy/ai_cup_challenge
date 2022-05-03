import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_energy_data() -> pd.DataFrame:
    file_path = "../data/start/train.csv"
    df = pd.read_csv(file_path)
    return df


def save_dataframe_as_pickle(df: pd.DataFrame):
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(df, f)


def load_dataframe_from_pickle() -> pd.DataFrame:
    with open('train_data.pkl', 'rb') as f:
        return pickle.load(f)


def calculate_daily_consumption(df: pd.DataFrame) -> pd.DataFrame:
    # 48 values represent one day
    daily = pd.DataFrame(index=df.index)
    date = "2017-01-01"
    for name, value in df.iteritems():
        current_date = name[0:10]
        if date == current_date:
            date = current_date
            if daily.columns.empty:
                daily[date] = 0
            if date in daily.columns:
                daily[date] += value.values
        else:
            date = current_date
            daily[date] = value.values
    pass


#data = load_energy_data()
#save_dataframe_as_pickle(data)
data = load_dataframe_from_pickle()
data = data.set_index("pseudo_id")
daily_data = calculate_daily_consumption(data)


print("hi")
