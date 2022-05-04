from enum import Enum
import tensorflow as tf


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
                 patience: int = -1):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.max_epochs = max_epochs
        self.patience = patience


ALL_MODELS = {
    "default": ModelParameter(),
    "patience": ModelParameter(patience=5)
}


class Setup:
    def __init__(self,
                 model_key="default",
                 normalization: Normalization = Normalization.MEAN,
                 n_ahead: int = 1,
                 n_before: int = 2,
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

        self.num_features = num_features

        self.normalization_name = f"normierung_{str(self.normalization.name).lower()}_"

        self.sliding_window_name = f"nahead_{str(self.n_ahead).lower()}_" \
                                   f"nbefore_{str(self.n_before).lower()}_" \
                                   f"datainterval_{str(self.data_interval).lower()}_"

        self.model_name = f"nahead_{str(self.n_ahead).lower()}_" \
                          f"nbefore_{str(self.n_before).lower()}_" \
                          f"model_{self.model_key}_"
