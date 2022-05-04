import os

# region Paths
from scripts.setup import ModelParameter, Setup, Normalization

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')

DIR_PREPROCESSING = f"{DIR_DATA}preprocessing"
DIR_SLIDING_WINDOW = f"{DIR_DATA}sliding_window"
DIR_VALIDATION = f"{DIR_DATA}validation"


ALL_SETUPS = {
    "default": Setup(),
    "normalization_zero_to_one": Setup(normalization=Normalization.ZERO_TO_ONE)
}

ACTUAL_KEY = "default"
ACTUAL_SETUP = ALL_SETUPS[ACTUAL_KEY]