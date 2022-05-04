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
    "patience": ModelParameter(patience=5)
}

ALL_SETUPS = {

}

ACTUAL_SETUP =


