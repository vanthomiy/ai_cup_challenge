import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras_visualizer import visualizer

import settings

setup = settings.Settings("hourly_mae_60_week")

model = tf.keras.models.load_model(setup.FILE_MODEL("all"))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# visualizer(model, format='png', view=True)
