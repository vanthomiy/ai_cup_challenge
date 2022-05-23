"""
We can use this to do automate test runs for a few setups at once
"""
import time

import settings
from scripts.preparation import Preparation
from scripts.train_model import TrainModel
from scripts.use_model_multiple_prediction import ModelMultiplePrediction

start_time = time.time()

for key in settings.ALL_SETUPS:
    try:
        setup = settings.Settings("mae_whithout_weather") #key)
        Preparation(setup).start()
        TrainModel(setup).start()
        ModelMultiplePrediction(setup).start()

    except Exception as ex:
        print("failed with " + str(key))

end_time = time.time()

print(start_time, end_time)
