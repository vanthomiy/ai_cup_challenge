import pickle
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import settings
import pandas as pd
from scripts.setup import Normalization, Timespan


def load_data() -> (pd.DataFrame, pd.DataFrame):
    """
    Load data from the 'train.csv' and 'counts.csv' csv files.

    :return: Two DataFrames which contain the data.
    """

    df1 = pd.read_csv(settings.FILE_REGULAR_TRAVEL, index_col='EZone')
    df2 = pd.read_csv(settings.FILE_REGULAR_ROUTE_DEFINITION, index_col='route')
    return df1, df2


def adjust_zone_name(train: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """
    Divide every value in the train DataFrame with the amount of dwellings fot the value in oder to get the average value for each cell.

    :param train: The train.csv dataset as a DataFrame, provided by the function load_data.
    :param counts: The counts.csv dataset as a DataFrame, provided by the function load_data.
    :return: The cleaned train dataset as DataFrame.
    """
    dfl = pd.DataFrame()

    zones = train.index.tolist()

    dfl["zones"] = [int(zone.split()[0]) for zone in zones]
    dfl.index = [datetime.strptime(row["date"] + " " + str(row["hour"]), '%Y-%m-%d %H') for index, row in
                 train.iterrows()]
    dfl["passangers"] = train["Passengers"].tolist()

    return dfl


def normalize_data_NONE():
    """
    Normalization algorithm which is to be defined.

    :return: tbd.
    """
    raise NotImplemented("Not Implemented yet.")


def normalize_data_MEAN(train_transposed: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Normalization which uses the MEAN algorithm.

    :param train_transposed: The train_cleaned dataset as a DataFrame, provided by the function clean_train_dataset.
    :return: df: The train_cleaned dataset as a copy and normalized.
             _normalization: The used normalization values as a dictionary.
    """

    df = train_transposed.copy()

    df = df.dropna(axis=0)

    # Calculate the normalization values and save them in a dictionary
    _normalization = {
        "mean": df["passangers"].mean(),
        "std": df["passangers"].std()
    }
    df["passangers"] = (df["passangers"] - _normalization["mean"]) / _normalization["std"]
    return df, _normalization


def normalize_data_ZERO_TO_ONE(train_transposed: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Normalization which uses the ZERO TO ONE algorithm.

    :param train_transposed: The train_cleaned dataset as a DataFrame, provided by the function clean_train_dataset.
    :return: df: The train_cleaned dataset as a copy and normalized.
             _normalization: The used normalization values as a dictionary.
    """

    df = train_transposed.copy()
    # Calculate the normalization values and save them in a dictionary
    _normalization = {
        "max": df.to_numpy().max(),
        "min": df.to_numpy().min()
    }
    df = (df - _normalization["min"]) / _normalization["max"]
    return df, _normalization


def save_normalization_values(normalization_values: dict):
    """
    Save the normalization values as pickle file.
    :param normalization_values:
    """

    with open(settings.FILE_NORMALIZATION_DATA, 'wb') as file:
        pickle.dump(normalization_values, file)


def split_dataset(df: pd.DataFrame) -> list:
    """
    Split the dataset at every point in time, where a gap is located. Get is references to as a lack of data provided by EON.

    :param df: The normalized dataset as a Dataframe, provided by one of the normalization functions.
    :return: A list containing the split dataset.
    """
    df_new = None

    # split dataset by busstop
    for bus_stop in settings.BUS_STOPS:
        df_stop = df[df["zones"] == int(bus_stop)]



        if df_new is None:
            index = df.index[df['zones'] == bus_stop].tolist()
            df_new = pd.DataFrame(index=index)

        df_new[bus_stop] = df_stop["passangers"]

    # Split factor which is defined by days * hours per day * data per hour / the amount of the data per interval tbu
    split_by = int(24 * 7 * 3 / settings.ACTUAL_SETUP.data_interval.value)
    temp_lst_split_dataset = []
    amount_of_times = len(df_new.index)
    # Loop through every timespan and split the dataset at the given index
    for i in range(0, int(amount_of_times / split_by)):
        temp_lst_split_dataset.append(
            df_new.iloc[i * split_by: (i + 1) * split_by]
        )
    return temp_lst_split_dataset


def create_and_save_data_windows(lst_split_dataset: list, _amplitude: int, _offset: int):
    """
    Create and save data windows with sin and cos values for both daily and annual ranges.

    :param lst_split_dataset: A list containing the split dataset provided by the function split_dataset.
    :param _amplitude: The amplitude used for the sin and cos functions.
    :param _offset: The offset used for the sin and cos functions.
    """

    # Calculate seconds per day for the sin and cos functions with 24 hours/day * 60 minutes/hour * 60 seconds/minute
    seconds_per_day = 24 * 60 * 60
    # Calculate seconds per day for the sin and cos functions with seconds/day * days/year
    seconds_per_year = 365.2425 * seconds_per_day
    # ID of the current window
    window_id = 0

    # Loop through every split timespan in the dataset
    for timespan in lst_split_dataset:
        # List which contains the data for every particular timespan
        lst_data_per_timestamp = []
        # Loop through every row in the current timespan dataframe
        for index, row in timespan.iterrows():
            # Parse the date from the rows time label
            # date_time = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
            # Get the timestamp in seconds from the date_time
            timestamp_s = int(round(index.timestamp()))
            # Create a dictionary which contains the daily and annual sin and cos values for the current timestamp
            dict_timestamp_data = {"day sin": _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_day)) + _offset,
                                   "day cos": _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_day)) + _offset,
                                   'year sin': _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_year)) + _offset,
                                   'year cos': _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_year)) + _offset}

            # Add the existing data for the current row and every id to the temporary data dictionary
            for bus_stop in settings.BUS_STOPS:
                dict_timestamp_data[bus_stop] = row[bus_stop]

            for weather_data in settings.ACTUAL_SETUP.weather_features:
                dict_timestamp_data[weather_data] = row[weather_data]

            # Save temporary dataframe in the list
            lst_data_per_timestamp.append(dict_timestamp_data)

        # Create a dataframe from the list which contains every
        df = pd.DataFrame(lst_data_per_timestamp)

        df["time"] = timespan.index
        # Save the dataframe to a csv file with the current id
        df.to_csv(settings.FILE_TIME_WINDOW_X(window_id))
        window_id += 1


def adjust_time_interval(df_t):
    """
    Adjusts the time interval of the dataframe.
    For 0-23 Hours-Intervals the data will be cut off
    For 24 Hour-Interval the data will be summed up
    :param df_t: The actual, already transposed, df
    :return: The intervall adjusted df
    """
    interval = settings.ACTUAL_SETUP.data_interval.value
    df_t_i = None
    if settings.ACTUAL_SETUP.data_interval != Timespan.DAILY:
        # Just cut the data
        df_t_i = df_t.iloc[::interval, :]
    else:
        # Sum up the data for the timespan
        # Maybe there is a faster way to do this
        # Creates emtpy df with columns
        df_t_i = pd.DataFrame(columns=df_t.columns.tolist())
        # Loops through the df and append n rows as sum to the new df
        for index in range(0, len(df_t.index), interval):
            df_t_i = df_t_i.append(df_t.iloc[index:index + interval].sum(), ignore_index=True)
        # loads the index (datetime)
        index = df_t.iloc[::interval, :].index
        # appends index to the created df
        df_t_i = df_t_i.set_index(index)

    return df_t_i


def save_normalization_plot(df_n):
    df_melt = df_n.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_melt)
    _ = ax.set_xticklabels(df_n.keys(), rotation=90)
    plt.savefig(settings.FILE_NORMALIZATION_PLOT)


def add_weather_data(df):
    # we define the wheater columns to use in the setup
    if len(settings.ACTUAL_SETUP.weather_features) <= 0:
        return df

    # load the weather data
    weather_df = pd.read_csv(settings.FILE_WEATHER_DATA)

    # make the dataset smaller. only columns and rows that are needed
    # exclude columns

    all_days = [str(datetime.strptime(string_day, '%Y-%m-%d %H:%M:%S').date()) for string_day in df.index.tolist()]
    all_days = list(dict.fromkeys(all_days))

    # condition mask
    mask = weather_df['time'].isin(all_days)

    # new dataframe with selected rows
    df_wr = pd.DataFrame(weather_df[mask])

    # norm the data
    mean = df_wr.mean()
    std = df_wr.std()

    df_wr = (df_wr - mean) / std

    n_times = int(24 / (settings.ACTUAL_SETUP.data_interval.value / 2))
    for col in settings.ACTUAL_SETUP.weather_features:
        col_data = []

        for data in df_wr[col].values:
            col_data.extend([data] * n_times)
        df[col] = col_data

    normalization_values = {"mean": mean, "std": std}

    with open(settings.FILE_NORMALIZATION_DATA_WEATHER, 'wb') as file:
        pickle.dump(normalization_values, file)

    return df
    # exclude all rows that are not necessary


# Load the data from the csv files
train_df, bus_stops_df = load_data()

# Clean the data with the amount of dwelling per id
train_df_cleaned = adjust_zone_name(train=train_df, counts=bus_stops_df)

# Adjust the time intervall of the data [half-hourly, hourly and daily]
train_df_cleaned_transposed_interval = adjust_time_interval(train_df_cleaned)

# Introduce necessary variables and predefine them as None
amplitude = None
offset = None
train_df_normalized = None
normalization = None

# Choose normalization method
if settings.ACTUAL_SETUP.normalization == Normalization.NONE:
    train_df_normalized = normalize_data_NONE()
elif settings.ACTUAL_SETUP.normalization == Normalization.MEAN:
    train_df_normalized, normalization = normalize_data_MEAN(train_transposed=train_df_cleaned_transposed_interval)
    amplitude = 2
    offset = 0
elif settings.ACTUAL_SETUP.normalization == Normalization.ZERO_TO_ONE:
    train_df_normalized, normalization = normalize_data_ZERO_TO_ONE(
        train_transposed=train_df_cleaned_transposed_interval)
    amplitude = 0.5
    offset = 0.5

# Check if anything went wrong
if amplitude is None or offset is None or train_df_normalized is None or normalization is None:
    raise ValueError("There is a value missing.")

# Save normalization values
save_normalization_values(normalization_values=normalization)

# Save the plot for the normalized data
save_normalization_plot(train_df_normalized)

# Add weather data if necessary
train_df_normalized_wd = add_weather_data(train_df_normalized)

# Split dataset at the gaps in the dataset
lst_split_df = split_dataset(df=train_df_normalized_wd)

# Create windows
create_and_save_data_windows(lst_split_dataset=lst_split_df, _amplitude=amplitude, _offset=offset)

print("Preparation script executed successfully.")
