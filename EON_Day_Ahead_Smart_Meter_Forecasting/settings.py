import os

# region Paths
import pandas as pd

from scripts.setup import Setup, Normalization

"""
    "mae_whithout_weather": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mae"),
    "mape_whithout_weather": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mape"),

    "mae_whithout_weather_4days": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 4, model_key="mae"),
    "mape_whithout_weather_4days": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 4, model_key="mape"),

    "mae_whithout_weather_5days": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 5, model_key="mae"),
    "mape_whithout_weather_5days": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 5, model_key="mape"),

    "mae_whithout_weather_half_prediction": Setup(pseudo_id_to_use=3, n_ahead=12, n_before=24 * 3, model_key="mae"),
    "mape_whithout_weather_half_prediction": Setup(pseudo_id_to_use=3, n_ahead=12, n_before=24 * 3, model_key="mape"),

    "mae_whit_temp": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mae",
                           weather_features=["tavg_mean"]),
    "mape_whit_temp": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mape",
                            weather_features=["tavg_mean"]),

    "mae_whith_mean_weather": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mae",
                                    weather_features=["tavg_mean", "prcp_mean", "snow_mean", "tsun_mean"]),
    "mape_whith_mean_weather": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mape",
                                     weather_features=["tavg_mean", "prcp_mean", "snow_mean", "tsun_mean"]),

    "mae_whith_mean_weather_more": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mae",
                                         weather_features=["tavg_mean", "prcp_mean", "snow_mean", "tsun_mean",
                                                           "pres_mean", "wpgt_mean", "wspd_mean"]),
    "mape_whith_mean_weather_more": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mape",
                                          weather_features=["tavg_mean", "prcp_mean", "snow_mean", "tsun_mean",
                                                            "pres_mean", "wpgt_mean", "wspd_mean"]),
    "mape_whithout_weather_mape_units_12": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3,
                                                 model_key="mape_units_12"),
    "mape_whithout_weather_week_forward": Setup(pseudo_id_to_use=3, n_ahead=24 * 7, n_before=24 * 4,
                                                model_key="mape"),
    "zero_one_mape_whithout_weather": Setup(normalization=Normalization.ZERO_TO_ONE, pseudo_id_to_use=3, n_ahead=24, n_before=24 * 3, model_key="mape"),
"""
ALL_SETUPS = {
    "mape_whit_temp_4days": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 4, model_key="mape",
                                         weather_features=["tavg_mean"]),
    "mape_whithout_weather_4days": Setup(pseudo_id_to_use=3, n_ahead=24, n_before=24 * 4, model_key="mape"),

}


class Settings:
    def __init__(self,
                 SETUP_KEY="mae_whithout_weather"):
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
        self.FILE_TRAIN_DATA = os.path.join(self.DIR_START, "train.csv")

        # File path of the original 'counts.csv' dataset which contains the amount of dwellings per id in the 'train.csv'
        self.FILE_COUNTS_DATA = os.path.join(self.DIR_START, "counts.csv")

        self.FILE_WEATHER_DATA = os.path.join(self.DIR_START, "de-weather-data-aggregated.csv")

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

        self.PSEUDO_IDS = pd.read_csv(self.FILE_TRAIN_DATA)["pseudo_id"].tolist()[:self.ACTUAL_SETUP.pseudo_id_to_use]

        self.FILE_SUBMISSION_NORMED_DATA = os.path.join(self.DIR_SUBMISSION, f"{SETUP_KEY}submission_normed.csv")
        self.FILE_SUBMISSION_DATA = os.path.join(self.DIR_SUBMISSION, f"{SETUP_KEY}submission_hourly.csv")
        self.FILE_SUBMISSION_DATA_DAILY = os.path.join(self.DIR_SUBMISSION, f"{SETUP_KEY}submission_daily.csv")

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
