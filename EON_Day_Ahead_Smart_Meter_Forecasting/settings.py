import os

# region Paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIR_DATA = os.path.join(PROJECT_ROOT, 'data\\')
DIR_MODEL = os.path.join(PROJECT_ROOT, 'model\\')
DIR_SCRIPTS = os.path.join(PROJECT_ROOT, 'scripts\\')
DIR_VALIDATION = os.path.join(PROJECT_ROOT, 'validation\\')

DIR_PREPROCESSING = f"{DIR_DATA}preprocessing"
DIR_SLIDING_WINDOW = f"{DIR_DATA}sliding_window"

ALL_MODELS = []

ALL_SETUPS = []

ACTUAL_SETUP =


