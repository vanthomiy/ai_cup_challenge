"""
We can use this to do automate test runs for a few setups at once
"""
import time

import settings
from scripts import preparation, sliding_window, train_model, evaluate_model, use_model_multiple_prediction, \
    evaluate_predictions
from scripts.evaluate_model import EvaluateModel
from scripts.evaluate_predictions import EvaluatePredictions
from scripts.preparation import Preparation
from scripts.sliding_window import Windowing
from scripts.train_model import TrainModel
from scripts.use_model_multiple_prediction import ModelMultiplePrediction

start_time = time.time()

for key in settings.ALL_SETUPS:
    try:
        setup = settings.Settings(key)
        # Preparation(setup).start()
        # preparation = Preparation(setup)
        # Windowing(setup).start()
        # TrainModel(setup).start()
        EvaluateModel(setup).start()
        ModelMultiplePrediction(setup).start()
        EvaluatePredictions(setup).start()

    except Exception as ex:
        print("failed with " + str(key))

end_time = time.time()

print(start_time, end_time)
