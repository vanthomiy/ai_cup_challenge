"""
We can use this to do automate test run
"""

import settings
from scripts.preparation import Preparation
from scripts.train_model import TrainModel
from scripts.use_model_multiple_prediction import ModelMultiplePrediction

setup = settings.Settings("test_lstm")
Preparation(setup).start()
TrainModel(setup).start()
ModelMultiplePrediction(setup).start()
