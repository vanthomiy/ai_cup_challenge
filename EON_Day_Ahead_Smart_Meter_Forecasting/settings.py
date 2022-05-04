import os

# region Paths
from scripts.setup import ModelParameter, Setup, Normalization

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')

DIR_START = os.path.join(DIR_DATA, 'Start\\')
DIR_PREPROCESSING = os.path.join(DIR_DATA, 'preprocessing\\')
DIR_SLIDING_WINDOW = os.path.join(DIR_DATA, 'sliding_window\\')
DIR_VALIDATION = os.path.join(DIR_DATA, 'validation\\')

SETUP_KEY = "default"

ALL_SETUPS = {
    "default": Setup(),
    "normalization_zero_to_one": Setup(normalization=Normalization.ZERO_TO_ONE)
}

ACTUAL_SETUP = ALL_SETUPS[SETUP_KEY]

# File path of the original 'train.csv' dataset which contains the kWh amounts per timespan and id
FILE_TRAIN_DATA = os.path.join(DIR_START, "train.csv")

# File path of the original 'counts.csv' dataset which contains the amount of dwellings per id in the 'train.csv'
FILE_COUNTS_DATA = os.path.join(DIR_START, "counts.csv")


# Filepath to the window by index and actual setup
def FILE_TIME_WINDOW_X(index: int):
    return os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}dataset{index}.csv")


# Filepath to the normalized values
FILE_NORMALIZATION_DATA = os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}normalized_values.csv")

# Filepath to the windowed data values
FILE_WINDOWED_DATA = os.path.join(DIR_SLIDING_WINDOW, f"{ACTUAL_SETUP.sliding_window_name}windowed_data.pkl")

# Filepath to the model
FILE_MODEL = os.path.join(DIR_MODEL, f"{ACTUAL_SETUP.model_name}model")


