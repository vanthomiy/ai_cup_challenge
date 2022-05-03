"""
This is the init script

We use the settings which defines the general data

Steps:
1. Load a setup
2. Load data from csv
3. Adjust data set time with [adjust_dataset_time]
4. Create features and norm the data [create_norm_features_data_set]
5. Train the model and save it
6. Test the model
7. Store validation data for the model
"""
import settings
from scripts.dataset import DatasetHandler
from scripts.model_setup import ModelSetup, Adjust


def build_pipeline(model: ModelSetup, dataset):
    """
    Build new models and save them for a setup
    Also create validation data
    :param model:
    :param dataset:
    :return:
    """
    # 3.
    adjusted_dataset = model.adjust_dataset_time(dataset)
    model.display_dataset(adjusted_dataset)
    # 4.
    normed_features_dataset = model.create_norm_features_data_set(adjusted_dataset)
    model.display_dataset(normed_features_dataset)
    # 5.
    model.train_model(normed_features_dataset)
    # 6.
    validation = model.test_model()
    # 7.
    model.display_metrics(validation)


def validation_pipeline(model: ModelSetup, dataset):
    """
    Load models and create validation data
    :param model:
    :param dataset:
    :return:
    """
    # 3.
    adjusted_dataset = model.adjust_dataset_time(dataset)
    model.display_dataset(adjusted_dataset)
    # 4.
    normed_features_dataset = model.create_norm_features_data_set(adjusted_dataset)
    model.display_dataset(normed_features_dataset)
    # 5.
    model.load_model()
    # 6.
    validation = model.test_model()
    # 7.
    model.display_metrics(validation)


def comparing_pipeline(models):
    """
    Compare all models in the list with each other and display metrics
    :param models:
    :return:
    """
    for model in models:
        pass


def display_data_pipeline(model, dataset):
    # 1.
    adjusted_dataset = model.adjust_dataset_time(dataset)
    model.display_dataset(adjusted_dataset)
    # 2.
    model.display_dataset(adjusted_dataset)


# Starting point

# Load the dataset
dataset = DatasetHandler(settings.DIR_DATA + "/start/train.csv")
data = dataset.load_features_data()
#dataset.plot_data(data)

setup = ModelSetup(60, Adjust.CUT, previous_data_for_forecast=12)
data1 = setup.adjust_dataset_time(data)
#dataset.plot_data(data1)

split_data = setup.split_train_validation_test_data(data1)
dataset_normed, train_normed = setup.create_norm_features_data_set(split_data)
setup.train_model(dataset_normed, load_model=False)

asd = 1
#for setup in settings.setups:
#    build_pipeline(setup, dataset.copy())
