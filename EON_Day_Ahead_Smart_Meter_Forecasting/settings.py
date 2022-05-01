import os
from scripts.model_setup import ModelSetup, Adjust

# region Paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')

# region Setups
time_intervals = {"half_hourly": 30, "hourly": 60, "daily": 60 * 24}
previous_data_intervals = [2, 3, 10]
setups = {}

for interval_key, interval_value in time_intervals.items():
    for interval_forcast_key, interval_forecast_value in time_intervals.items():
        for prev_interval in previous_data_intervals:
            for adjustment in Adjust:
                setups[interval_key + "_" + interval_forcast_key + "_" + str(prev_interval) + "_" + str(adjustment)] = \
                    ModelSetup(
                        dataset_time_interval=interval_value,
                        forecast_next_n_minutes=interval_forecast_value,
                        previous_data_for_forecast=prev_interval,
                        dataset_time_adjustment=adjustment)



