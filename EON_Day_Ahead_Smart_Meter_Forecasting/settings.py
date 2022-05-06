import os

# region Paths
import pandas as pd

from scripts.setup import ModelParameter, Setup, Normalization

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')

DIR_START = os.path.join(DIR_DATA, 'start\\')
DIR_PREPROCESSING = os.path.join(DIR_DATA, 'preprocessing\\')
DIR_SLIDING_WINDOW = os.path.join(DIR_DATA, 'sliding_window\\')
DIR_VALIDATION = os.path.join(DIR_DATA, 'validation\\')

TEST_TRAIN_VALID = ["train", "test", "val"]

SETUP_KEY = "fast_lane1"

ALL_SETUPS = {
    "default": Setup(),
    "fast_lane": Setup(pseudo_id_to_use=1, time_windows_to_use=1, model_key="fast_lane"),
    "fast_lane1": Setup(pseudo_id_to_use=20, time_windows_to_use=2, model_key="fast_lane")
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

# Filepath to normalization plot
FILE_NORMALIZATION_PLOT = os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}normalized_plot.png")


# Filepath to the windowed data values
def FILE_WINDOWED_DATA(index: str):
    return os.path.join(DIR_SLIDING_WINDOW, f"{ACTUAL_SETUP.sliding_window_name}windowed_data_{index}.pkl")


# Filepath to the model
FILE_MODEL = os.path.join(DIR_MODEL, f"{ACTUAL_SETUP.model_name}model")

FILE_MODEL_TRAIN = os.path.join(DIR_MODEL, f"{ACTUAL_SETUP.model_name}train.png")

FILE_EVALUATION_DATA = os.path.join(DIR_VALIDATION, f"evaluation.csv")

FILE_EVALUATION_OVERVIEW = os.path.join(DIR_VALIDATION, f"evaluation.png")

FILE_EVALUATION_TIMESERIES = os.path.join(DIR_VALIDATION, f"{ACTUAL_SETUP.sliding_window_name}timeseries.png")

PSEUDO_IDS = pd.read_csv(FILE_TRAIN_DATA)["pseudo_id"].tolist()[:ACTUAL_SETUP.pseudo_id_to_use]
