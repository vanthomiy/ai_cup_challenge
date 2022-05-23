"""
We can use this to do automate test run
"""
import time

import settings
from scripts.evaluate_model import EvaluateModel
from scripts.evaluate_predictions import EvaluatePredictions
from scripts.preparation import Preparation
from scripts.sliding_window import Windowing
from scripts.train_model import TrainModel
from scripts.use_model_multiple_prediction import ModelMultiplePrediction

setup = settings.Settings("mape_whithout_weather_1days_30")
Preparation(setup).start()
Windowing(setup).start()
TrainModel(setup).start()
EvaluateModel(setup).start()
ModelMultiplePrediction(setup).start()
EvaluatePredictions(setup).start()