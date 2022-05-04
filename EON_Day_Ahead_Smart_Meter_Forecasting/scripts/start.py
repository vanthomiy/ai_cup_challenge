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


# Starting point
recalculate_data = False

setup = ModelSetup(60, Adjust.CUT, previous_data_for_forecast=24, max_epochs=100)

if recalculate_data:
    # Load the dataset
    dataset = DatasetHandler(settings.DIR_DATA + "/fake_start/fake_train.csv")
    data = dataset.load_dataset_and_create_features()
    # data = dataset.load_features_data()
    # dataset.plot_data(data)

    data1 = setup.adjust_dataset_time(data)
    # dataset.plot_data(data1)

    split_data = setup.split_train_validation_test_data(data1)
    # dataset_normed = setup.create_norm_features_data_set(split_data)
    dataset_reordered = setup.do_murks(split_data)
    setup.train_model(dataset_reordered, load_model=False)
else:
    setup.train_model(load_model=False)
    setup.display_accuracy()

asd = 1
#for setup in settings.setups:
#    build_pipeline(setup, dataset.copy())
