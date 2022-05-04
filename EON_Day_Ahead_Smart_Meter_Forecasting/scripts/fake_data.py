from datetime import datetime
import numpy as np
import settings
import pandas as pd


original_dataset = pd.read_csv(f"{settings.DIR_DATA}start/train.csv", index_col='pseudo_id')
sliced_dataset = original_dataset.iloc[:1]
fake_dataset = sliced_dataset.copy()

day = 13 * 45 * 63

for col in sliced_dataset.columns:
    date_time = datetime.strptime(col, '%Y-%m-%d %H:%M:%S')
    timestamp_s = int(round(date_time.timestamp()))
    fake_dataset[col] = np.sin(timestamp_s * (np.pi / day))

fake_dataset.to_csv(f"{settings.DIR_DATA}fake_start/fake_train.csv")