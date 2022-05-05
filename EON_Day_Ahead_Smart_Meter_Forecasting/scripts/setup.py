from enum import Enum
import tensorflow as tf


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
    HALF_HOURLY = 30
    HOURLY = 60
    DAILY = 24 * 60


class ModelParameter:
    def __init__(self,
                 loss=tf.losses.MeanSquaredError(),
                 optimizer=tf.optimizers.Adam(),
                 metrics=[tf.metrics.MeanAbsolutePercentageError()],
                 max_epochs: int = 100,
                 patience: int = 100,
                 algorithm: Algorithm = Algorithm.LSTM):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.max_epochs = max_epochs
        self.patience = patience
        self.algorithm = algorithm


ALL_MODELS = {
    "default": ModelParameter(),
    "fast_lane": ModelParameter(max_epochs=3),
    "patience": ModelParameter(patience=5)
}


class Setup:
    def __init__(self,
                 model_key="default",
                 normalization: Normalization = Normalization.MEAN,
                 n_ahead: int = 1,
                 n_before: int = 1,
                 time_windows_to_use=21,
                 pseudo_id_to_use=60,
                 features=["day sin", "day cos", "year sin", "year cos"],
                 num_features=6,
                 data_interval: Timespan = Timespan.HOURLY):
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
        self.num_features = num_features
        """The count of the features"""
        self.time_windows_to_use = time_windows_to_use
        """How much of the 21 data windows should be used"""
        self.pseudo_id_to_use = pseudo_id_to_use
        """How much of the households should be used (value between 1 and 60)"""
        self.normalization_name = f"normierung_{str(self.normalization.name).lower()}_"

        self.sliding_window_name = f"{self.normalization_name}_" \
                                   f"nahead_{str(self.n_ahead).lower()}_" \
                                   f"nbefore_{str(self.n_before).lower()}_" \
                                   f"datainterval_{str(self.data_interval).lower()}_"

        self.model_name = f"{self.sliding_window_name}_" \
                          f"model_{self.model_key}_"
