import settings
import pandas as pd

from scripts.setup import Normalization


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
    assert False, "Not implemented yet."


def normalize_data_MEAN(train_transposed: pd.DataFrame):
    df = train_transposed.copy()
    _amplitude = 2
    _offset = 0
    normalization = {
        "mean": df.to_numpy().mean(),
        "std": df.to_numpy().std()
    }
    df = (df - normalization["std"]) / normalization["mean"]
    return df, _amplitude, _offset


def normalize_data_ZERO_TO_ONE(train_transposed: pd.DataFrame):
    df = train_transposed.copy()
    _amplitude = 0.5
    _offset = 0.5
    normalization = {
        "max": df.to_numpy().max(),
        "min": df.to_numpy().min()
    }

    df = (df - normalization["min"]) / normalization["max"]
    return df, _amplitude, _offset


# Load the data from the csv files
train_df, counts_df = load_data()
# Clean the data with the amount of dwelling per id
train_df_cleaned = clean_train_dataset(train=train_df, counts=counts_df)
# Test the cleaning function
test_clean_train_dataset(train_df, train_df_cleaned)
# Transpose the dataset for the next steps
train_df_cleaned_transposed = train_df_cleaned.T

# Choose the normalization function
if settings.ACTUAL_SETUP.normalization == Normalization.NONE:
    train_df_normalized = normalize_data_NONE()
elif settings.ACTUAL_SETUP.normalization == Normalization.MEAN:
    train_df_normalized, amplitude, offset = normalize_data_MEAN(train_df_cleaned_transposed)
elif settings.ACTUAL_SETUP.normalization == Normalization.ZERO_TO_ONE:
    train_df_normalized, amplitude, offset = normalize_data_ZERO_TO_ONE(train_df_cleaned_transposed)

print("hi")
