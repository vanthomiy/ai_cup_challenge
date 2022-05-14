"""
We can use this to do automate test runs for a few setups at once
"""
import time

import settings
from scripts import preparation, sliding_window, train_model, evaluate_model, use_model_multiple_prediction, \
    evaluate_predictions

start_time = time.time()

for key in settings.ALL_SETUPS:
    settings.ACTUAL_SETUP = settings.ALL_SETUPS[key]

    preparation.start()
    sliding_window.start()
    train_model.start()
    evaluate_model.start()
    use_model_multiple_prediction.start()
    evaluate_predictions.start()

end_time = time.time()

print(start_time, end_time)
