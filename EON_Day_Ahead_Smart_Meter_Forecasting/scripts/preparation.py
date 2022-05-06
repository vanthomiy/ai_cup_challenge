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

    df1 = pd.read_csv(settings.FILE_TRAIN_DATA, index_col='pseudo_id')
    df2 = pd.read_csv(settings.FILE_COUNTS_DATA, index_col='pseudo_id')
    return df1, df2


def clean_train_dataset(train: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """
    Divide every value in the train DataFrame with the amount of dwellings fot the value in oder to get the average value for each cell.

    :param train: The train.csv dataset as a DataFrame, provided by the function load_data.
    :param counts: The counts.csv dataset as a DataFrame, provided by the function load_data.
    :return: The cleaned train dataset as DataFrame.
    """

    df = train.copy()
    # Loop though every row of the train DataFrame
    for index, _ in df.iterrows():
        # Get the amount of dwellings for the current row index
        factor = counts.loc[index]['n_dwellings']
        # Divide the values of the train row by the factor
        df.loc[index] = df.loc[index].div(factor)
    return df


def test_clean_train_dataset(train: pd.DataFrame, cleaned: pd.DataFrame):
    """
    Test if the division performed in the function clean_train_dataset is performed properly.

    :param train: The train.csv dataset as a DataFrame, provided by the function load_data.
    :param cleaned: The train_cleaned dataset as a DataFrame, provided by the function clean_train_dataset.
    """

    id1 = "0x16cb02173ebf3059efdc97fd1819f14a2"
    id3 = "0x1612e4cbe3b1b85c3dbcaeaa504ee8424"
    factor_id1 = 288
    factor_id3 = 38

    train_sum_row1 = train.loc[id1].sum()
    cleaned_sum_row1 = cleaned.loc[id1].sum()
    cleaned_sum_row1 = cleaned_sum_row1 * factor_id1

    train_sum_row3 = train.loc[id3].sum()
    cleaned_sum_row3 = cleaned.loc[id3].sum()
    cleaned_sum_row3 = cleaned_sum_row3 * factor_id3

    assert train_sum_row1 == cleaned_sum_row1, "There is something wrong in clean_train_dataset"
    assert train_sum_row3 == cleaned_sum_row3, "There is something wrong in clean_train_dataset"


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

    # Calculate the normalization values and save them in a dictionary
    _normalization = {
        "mean": df.to_numpy().mean(),
        "std": df.to_numpy().std()
    }
    df = (df - _normalization["mean"]) / _normalization["std"]
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

    # Split factor which is defined by days * hours per day * data per hour
    split_by = 38 * 24 * 2
    temp_lst_split_dataset = []
    amount_of_times = len(df.index)
    # Loop through every timespan and split the dataset at the given index
    for i in range(0, int(amount_of_times / split_by)):
        temp_lst_split_dataset.append(
            df.iloc[i * split_by: (i + 1) * split_by]
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
            date_time = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
            # Get the timestamp in seconds from the date_time
            timestamp_s = int(round(date_time.timestamp()))
            # Create a dictionary which contains the daily and annual sin and cos values for the current timestamp
            dict_timestamp_data = {"day sin": _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_day)) + _offset,
                                   "day cos": _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_day)) + _offset,
                                   'year sin': _amplitude * np.sin(timestamp_s * (np.pi / seconds_per_year)) + _offset,
                                   'year cos': _amplitude * np.cos(timestamp_s * (np.pi / seconds_per_year)) + _offset}

            # Add the existing data for the current row and every id to the temporary data dictionary
            for pseudo_id in settings.PSEUDO_IDS:
                dict_timestamp_data[pseudo_id] = row[pseudo_id]

            # Save temporary dataframe in the list
            lst_data_per_timestamp.append(dict_timestamp_data)

        # Create a dataframe from the list which contains every
        df = pd.DataFrame(lst_data_per_timestamp)
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


# Load the data from the csv files
train_df, counts_df = load_data()

# Clean the data with the amount of dwelling per id
train_df_cleaned = clean_train_dataset(train=train_df, counts=counts_df)

# Test the cleaning function
test_clean_train_dataset(train=train_df, cleaned=train_df_cleaned)

# Transpose the dataset for the next steps
train_df_cleaned_transposed = train_df_cleaned.T

# Adjust the time intervall of the data [half-hourly, hourly and daily]
train_df_cleaned_transposed_interval = adjust_time_interval(train_df_cleaned_transposed)

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
    train_df_normalized, normalization = normalize_data_ZERO_TO_ONE(train_transposed=train_df_cleaned_transposed_interval)
    amplitude = 0.5
    offset = 0.5

# Check if anything went wrong
if amplitude is None or offset is None or train_df_normalized is None or normalization is None:
    raise ValueError("There is a value missing.")

# Save normalization values
save_normalization_values(normalization_values=normalization)

# Save the plot for the normalized data
save_normalization_plot(train_df_normalized)

# Split dataset at the gaps in the dataset
lst_split_df = split_dataset(df=train_df_normalized)

# Create windows
create_and_save_data_windows(lst_split_dataset=lst_split_df, _amplitude=amplitude, _offset=offset)

print("Preparation script executed successfully.")
