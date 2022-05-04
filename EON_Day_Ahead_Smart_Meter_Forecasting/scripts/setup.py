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
    def __int__(self,
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

    def name(self):
        return f"ml{self.loss}_mo{self.optimizer}_mm{self.metrics}_me{self.max_epochs}_mp{self.patience}_"


class Setup:
    def __init__(self, normalization: Normalization, n_ahead: int, n_before: int, data_interval: Timespan,
                 model_parameters: ModelParameter):
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
        self.model_parameters = model_parameters
        """The model parameters from the model class"""

    def name(self):
        first = f"sn{self.normalization}_sa{self.n_ahead}_sb{self.n_before}_sw{self.window_size}_mi{self.data_interval}_"
        return first + self.model_parameters.name()
