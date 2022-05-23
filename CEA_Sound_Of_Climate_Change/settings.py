import os

# region Paths
import pandas as pd

from scripts.setup import Setup, Normalization


ALL_SETUPS = {

    "test_lstm": Setup(n_ahead=24, n_before=24 * 1, model_key="lstm"),
    "test_transformer": Setup(n_ahead=24, n_before=24 * 1, model_key="transformer"),

}


class Settings:
    def __init__(self,
                 SETUP_KEY="test_transformer"):
        self.SETUP_KEY = SETUP_KEY
        self.PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
        self.DIR_DATA = os.path.join(self.PROJECT_ROOT, 'data\\')
        self.DIR_MODEL = os.path.join(self.PROJECT_ROOT, 'model\\')
        self.DIR_SCRIPTS = os.path.join(self.PROJECT_ROOT, 'scripts\\')

        self.DIR_START = os.path.join(self.DIR_DATA, 'start\\')
        self.DIR_PREPROCESSING = os.path.join(self.DIR_DATA, 'preprocessing\\')
        self.DIR_SLIDING_WINDOW = os.path.join(self.DIR_DATA, 'sliding_window\\')
        self.DIR_VALIDATION = os.path.join(self.DIR_DATA, 'validation\\')
        self.DIR_SUBMISSION = os.path.join(self.DIR_DATA, 'submission\\')

        self.TEST_TRAIN_VALID = ["train", "test", "val"]

        self.ACTUAL_SETUP = ALL_SETUPS[self.SETUP_KEY]

        # File path of the original 'train.csv' dataset which contains the kWh amounts per timespan and id
        self.FILE_TRAIN_DATA = os.path.join(self.DIR_START, "IAsignals_train_v2.mat")

        # File path of the original 'counts.csv' dataset which contains the amount of dwellings per id in the 'train.csv'
        self.FILE_TEST_DATA = os.path.join(self.DIR_START, "IAsignals_test_predictors.mat")

        self.FILE_PRECESSED_DATA_TRAIN = os.path.join(self.DIR_PREPROCESSING, f"{self.SETUP_KEY}train.csv")
        self.FILE_PRECESSED_DATA_TEST = os.path.join(self.DIR_PREPROCESSING, f"{self.SETUP_KEY}test.csv")


        # Filepath to the normalized values
        self.FILE_NORMALIZATION_DATA = os.path.join(self.DIR_PREPROCESSING, f"{SETUP_KEY}normalized_values.pkl")
        self.FILE_NORMALIZATION_DATA_WEATHER = os.path.join(self.DIR_PREPROCESSING,
                                                            f"{SETUP_KEY}normalized_weather_values.pkl")
        # Filepath to normalization plot
        self.FILE_NORMALIZATION_PLOT = os.path.join(self.DIR_PREPROCESSING, f"{SETUP_KEY}normalized_plot.png")

        self.FILE_EVALUATION_DATA = os.path.join(self.DIR_VALIDATION, f"evaluation.csv")
        self.FILE_MAPE_EVALUATION_DATA = os.path.join(self.DIR_VALIDATION, f"mape_evaluation.csv")

        self.FILE_EVALUATION_OVERVIEW = os.path.join(self.DIR_VALIDATION, f"evaluation.png")
        self.FILE_MAPE_EVALUATION_OVERVIEW = os.path.join(self.DIR_VALIDATION, f"mape_evaluation.png")

        self.FILE_EVALUATION_TIMESERIES = os.path.join(self.DIR_VALIDATION, f"{SETUP_KEY}timeseries.png")
        self.FILE_MAPE_EVALUATION_TIMESERIES = os.path.join(self.DIR_VALIDATION, f"{SETUP_KEY}timeseries_mape.png")

        self.FILE_SUBMISSION_DATA = os.path.join(self.DIR_SUBMISSION, f"{SETUP_KEY}submission.csv")

        # Filepath to the model
        self.FILE_MODEL = os.path.join(self.DIR_MODEL, f"{SETUP_KEY}_model")

    # Filepath to the window by index and actual setup
    def FILE_TIME_WINDOW_X(self, index: int):
        return os.path.join(self.DIR_PREPROCESSING, f"{self.SETUP_KEY}dataset{index}.csv")

    # Filepath to the windowed data values
    def FILE_WINDOWED_DATA(self, index: str):
        return os.path.join(self.DIR_SLIDING_WINDOW, f"{self.SETUP_KEY}windowed_data_{index}.pkl")

    def FILE_MODEL_TRAIN(self, name: str):
        return os.path.join(self.DIR_MODEL, f"{self.SETUP_KEY + name}.png")
