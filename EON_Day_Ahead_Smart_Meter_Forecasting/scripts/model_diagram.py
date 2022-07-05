import tensorflow as tf
from keras.utils.vis_utils import plot_model

import settings

setup = settings.Settings("hourly_mae_5_week_fast_zero")

model = tf.keras.models.load_model(setup.FILE_MODEL("all"))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)