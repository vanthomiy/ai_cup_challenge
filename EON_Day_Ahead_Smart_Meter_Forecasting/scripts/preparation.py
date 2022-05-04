import settings
import pandas as pd


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


train_df, counts_df = load_data()
train_df_cleaned = clean_train_dataset(train=train_df, counts=counts_df)
test_clean_train_dataset(train_df, train_df_cleaned)

print("hi")
