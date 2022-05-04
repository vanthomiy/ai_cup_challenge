import settings
import pandas as pd


def load_datasets():
    # Load data from the corresponding csv files
    train_df = pd.read_csv(settings.FILE_TRAIN_DATA, index_col='pseudo_id')
    counts_df = pd.read_csv(settings.FILE_COUNTS_DATA, index_col='pseudo_id')

    # Divide every value in the train_df dataset with the amount of dwellings for the value in order to get the
    # average value for each cell
    for index, _ in train_df.iterrows():
        factor = counts_df.loc[index]['n_dwellings']
        train_df.loc[index] = train_df.loc[index].div(factor)

    train_df_transposed = train_df.T


load_datasets()