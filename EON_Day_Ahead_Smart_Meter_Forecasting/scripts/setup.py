from enum import Enum
import tensorflow as tf


class ID_HANDLING(Enum):
    SINGLE = 1
    MULTIPLE = 2
    ALL = 3


class Algorithm(Enum):
    LINEAR = 1
    DENSE = 2
    CONV = 3
    LSTM = 4


class Normalization(Enum):
    NONE = 1
    """No normalization"""
    MEAN = 2
    """Take mean/std around 0"""
    ZERO_TO_ONE = 3
    """Make values between 0 and 1"""


class Timespan(Enum):
    HALF_HOURLY = 1
    HOURLY = 2
    DAILY = 48


class ModelParameter:
    def __init__(self,
                 loss=tf.losses.MeanSquaredError(),
                 optimizer=tf.optimizers.Adam(),
                 metrics=[tf.metrics.MeanAbsoluteError()],
                 stop_loss="mean_absolute_error",
                 lstm_units=1,
                 max_epochs: int = 1000,
                 patience: int = 3,
                 algorithm: Algorithm = Algorithm.LSTM):
        self.loss = loss
        self.stop_loss = stop_loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.lstm_units = lstm_units
        self.max_epochs = max_epochs
        self.patience = patience
        self.algorithm = algorithm


ALL_MODELS = {
    "fast": ModelParameter(max_epochs=3,
                            stop_loss="mean_absolute_percentage_error",
                           metrics=[tf.metrics.MeanAbsolutePercentageError()]),
    "mape": ModelParameter(stop_loss="mean_absolute_percentage_error",
                           metrics=[tf.metrics.MeanAbsolutePercentageError()]),
    "mape_units_12": ModelParameter(stop_loss="mean_absolute_percentage_error",
                                    metrics=[tf.metrics.MeanAbsolutePercentageError()],
                                    lstm_units=12),
    "mape_optimizer_nadam": ModelParameter(stop_loss="mean_absolute_percentage_error",
                                           metrics=[tf.metrics.MeanAbsolutePercentageError()],
                                           optimizer=tf.optimizers.Nadam),
    "mae": ModelParameter(stop_loss="mean_absolute_error", metrics=[tf.metrics.MeanAbsoluteError()]),
}


class Setup:
    def __init__(self,
                 model_key="default_lstm",
                 normalization: Normalization = Normalization.MEAN,
                 n_ahead: int = 24,
                 n_before: int = 24,
                 time_windows_to_use=22,
                 pseudo_id_to_use=60,
                 features=["day sin", "day cos", "year sin", "year cos"],
                 # weather_features = ["tavg_mean","tavg_std","tmin_mean","tmin_std","tmax_mean","tmax_std","prcp_mean","prcp_std","snow_mean","snow_std","wdir_mean","wdir_std","wspd_mean","wspd_std","wpgt_mean","wpgt_std","pres_mean","pres_std","tsun_mean","tsun_std"],
                 weather_features=[],
                 # num_features=6,
                 id_handling: ID_HANDLING = ID_HANDLING.MULTIPLE,
                 data_interval: Timespan = Timespan.HOURLY):
        self.id_handling = id_handling

        self.normalization = normalization
        """What normalization should be used? \"none\", \"mean\", \"01\""""
        self.n_ahead = n_ahead
        """What time span should be predicted? In Minutes"""
        self.n_before = n_before
        """What time span should be used for the prediction? In Minutes"""
        self.window_size = n_ahead + n_before
        """The size of the whole window"""
        self.data_interval = data_interval
        """What time intervall should be used for the data? In Minutes"""
        self.model_parameters: ModelParameter = ALL_MODELS[model_key]
        """The model parameters from the model class"""
        self.model_key = model_key
        """The actual columns / features for the model"""
        self.features = features
        """The name of the features"""
        self.num_features = len(features) + 2 + len(weather_features) - (1 if id_handling != ID_HANDLING.MULTIPLE else 0)
        """The count of the features"""
        self.time_windows_to_use = time_windows_to_use
        """How much of the 21 data windows should be used"""
        self.pseudo_id_to_use = pseudo_id_to_use
        """How much of the households should be used (value between 1 and 60)"""
        self.normalization_name = f"normierung_{str(self.normalization.name).lower()}_" \
                                  f"fc_{str(self.num_features)}_"
        self.weather_features = weather_features
        self.sliding_window_name = f"{self.normalization_name}_" \
                                   f"nahead_{str(self.n_ahead).lower()}_" \
                                   f"nbefore_{str(self.n_before).lower()}_" \
                                   f"datainterval_{str(self.data_interval).lower()}_"

        self.model_name = f"{self.sliding_window_name}_" \
                          f"model_{self.model_key}_"
