import os

# region Paths
import pandas as pd

from scripts.setup import ModelParameter, Setup, Normalization, Timespan

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')

DIR_START = os.path.join(DIR_DATA, 'start\\')
DIR_PREPROCESSING = os.path.join(DIR_DATA, 'preprocessing\\')
DIR_SLIDING_WINDOW = os.path.join(DIR_DATA, 'sliding_window\\')
DIR_VALIDATION = os.path.join(DIR_DATA, 'validation\\')
DIR_SUBMISSION = os.path.join(DIR_DATA, 'submission\\')

TEST_TRAIN_VALID = ["train", "test", "val"]

# Change this key to use another setup
SETUP_KEY = "submission_daily"

ALL_SETUPS = {
    "daily_lstm": Setup(pseudo_id_to_use=1, n_ahead=24, n_before=24 * 3, model_key="default_lstm"),
    "daily_mape": Setup(pseudo_id_to_use=1, n_ahead=24, n_before=24 * 3, model_key="mape"),
    "daily_mape_weather": Setup(pseudo_id_to_use=1, n_ahead=24, n_before=24 * 3, model_key="mape",
                                weather_features=["tavg_mean","snow_mean","wspd_mean","tsun_mean"]),
    "daily_mape_week": Setup(pseudo_id_to_use=1, n_ahead=24, n_before=24 * 7, model_key="mape"),

    "submission_daily": Setup(pseudo_id_to_use=1, n_ahead=7, data_interval=Timespan.DAILY, n_before=7, model_key="mape",
                              weather_features=["tavg_mean","snow_mean","wspd_mean","tsun_mean"]),

    "submission_hourly": Setup(pseudo_id_to_use=60, n_ahead=24, n_before=24 * 3, model_key="mape",
                               weather_features=["tavg_mean","snow_mean","wspd_mean","tsun_mean"]),
}

ACTUAL_SETUP = ALL_SETUPS[SETUP_KEY]

# File path of the original 'train.csv' dataset which contains the kWh amounts per timespan and id
FILE_TRAIN_DATA = os.path.join(DIR_START, "train.csv")

# File path of the original 'counts.csv' dataset which contains the amount of dwellings per id in the 'train.csv'
FILE_COUNTS_DATA = os.path.join(DIR_START, "counts.csv")

FILE_WEATHER_DATA = os.path.join(DIR_START, "de-weather-data-aggregated.csv")

# Filepath to the window by index and actual setup
def FILE_TIME_WINDOW_X(index: int):
    return os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}dataset{index}.csv")



# Filepath to the normalized values
FILE_NORMALIZATION_DATA = os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}normalized_values.pkl")
FILE_NORMALIZATION_DATA_WEATHER = os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}normalized_weather_values.pkl")

# Filepath to normalization plot
FILE_NORMALIZATION_PLOT = os.path.join(DIR_PREPROCESSING, f"{ACTUAL_SETUP.normalization_name}normalized_plot.png")


# Filepath to the windowed data values
def FILE_WINDOWED_DATA(index: str):
    return os.path.join(DIR_SLIDING_WINDOW, f"{ACTUAL_SETUP.sliding_window_name}windowed_data_{index}.pkl")


# Filepath to the model
FILE_MODEL = os.path.join(DIR_MODEL, f"{ACTUAL_SETUP.model_name}model")


def FILE_MODEL_TRAIN(name: str):
    return os.path.join(DIR_MODEL, f"{ACTUAL_SETUP.model_name + name}.png")


FILE_EVALUATION_DATA = os.path.join(DIR_VALIDATION, f"evaluation.csv")
FILE_MAPE_EVALUATION_DATA = os.path.join(DIR_VALIDATION, f"mape_evaluation.csv")

FILE_EVALUATION_OVERVIEW = os.path.join(DIR_VALIDATION, f"evaluation.png")
FILE_MAPE_EVALUATION_OVERVIEW = os.path.join(DIR_VALIDATION, f"mape_evaluation.png")

FILE_EVALUATION_TIMESERIES = os.path.join(DIR_VALIDATION, f"{ACTUAL_SETUP.sliding_window_name}timeseries.png")
FILE_MAPE_EVALUATION_TIMESERIES = os.path.join(DIR_VALIDATION, f"{ACTUAL_SETUP.sliding_window_name}timeseries_mape.png")

PSEUDO_IDS = pd.read_csv(FILE_TRAIN_DATA)["pseudo_id"].tolist()[:ACTUAL_SETUP.pseudo_id_to_use]

FILE_SUBMISSION_NORMED_DATA = os.path.join(DIR_SUBMISSION, f"{ACTUAL_SETUP.sliding_window_name}submission_normed.csv")
FILE_SUBMISSION_DATA = os.path.join(DIR_SUBMISSION, f"{ACTUAL_SETUP.sliding_window_name}submission.csv")

