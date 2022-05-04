import os

# region Paths
from scripts.setup import ModelParameter

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')

DIR_PREPROCESSING = f"{DIR_DATA}preprocessing"
DIR_SLIDING_WINDOW = f"{DIR_DATA}sliding_window"
DIR_VALIDATION = f"{DIR_DATA}validation"

ALL_MODELS = {
    "default": ModelParameter(),
    "patience": ModelParameter(patience=1)
}

ALL_SETUPS = {

}

ACTUAL_SETUP = []

# File path of the original 'train.csv' dataset which contains the kWh amounts per timespan and id
FILE_TRAIN_DATA = os.path.join(PROJECT_ROOT, "data\\Start\\train.csv")

# File path of the original 'counts.csv' dataset which contains the amount of dwellings per id in the 'train.csv'
FILE_COUNTS_DATA = os.path.join(PROJECT_ROOT, "data\\Start\\counts.csv")
