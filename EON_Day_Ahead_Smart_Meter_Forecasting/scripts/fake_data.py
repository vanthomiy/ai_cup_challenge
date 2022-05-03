import settings
from scripts.dataset import DatasetHandler
from scripts.model_setup import ModelSetup, Adjust
import pandas as pd


original_dataset = pd.read_csv(f"{settings.DIR_DATA}start/train.csv", index_col='pseudo_id')
fake_dataset = original_dataset.iloc[:1]

fake_dataset.to_csv(f"{settings.DIR_DATA}fake_start/fake_train.csv")
print("hi")