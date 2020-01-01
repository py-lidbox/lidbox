"""
TensorFlow implementation of the X-vector DNN consisting of five temporal convolutions followed by stats pooling and 3 fully connected layers.
See also:
David Snyder, et al. "Spoken Language Recognition using X-vectors." Odyssey. 2018.
http://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
"""
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Layer,
    ReLU,
)
from tensorflow.keras.models import Sequential
import tensorflow as tf

from .tdnn import TDNN


xvector_layer_names = ["frame{:d}".format(i) for i in range(1, 6)] + ["stats_pooling", "segment6"]

class GlobalMeanStddevPooling1D(Layer):
    """Compute arithmetic mean and standard deviation of the inputs over the time dimension, then output the concatenation of the computed stats."""
    def call(self, inputs):
        # channels_last
        steps_axis = 1
        mean = tf.math.reduce_mean(inputs, axis=steps_axis)
        std = tf.math.reduce_std(inputs, axis=steps_axis)
        return tf.concat((mean, std), axis=1)

def loader(input_shape, num_outputs, output_activation="softmax", use_conv1d_for_tdnn=True):
    if use_conv1d_for_tdnn:
        tdnn = [
            Conv1D(512, 5, 1, padding="valid", activation="relu", name="frame1", input_shape=input_shape),
            Conv1D(512, 3, 2, padding="valid", activation="relu", name="frame2"),
            Conv1D(512, 3, 3, padding="valid", activation="relu", name="frame3"),
            Conv1D(512, 1, 1, padding="valid", activation="relu", name="frame4"),
            Conv1D(1500, 1, 1, padding="valid", activation="relu", name="frame5"),
        ]
    else:
        tdnn = [
            TDNN(512, [-2, -1, 0, 1, 2], activation="relu", name="frame1", input_shape=input_shape),
            TDNN(512, [-2, 0, 2], activation="relu", name="frame2"),
            TDNN(512, [-3, 0, 3], activation="relu", name="frame3"),
            TDNN(512, [0], activation="relu", name="frame4"),
            TDNN(1500, [0], activation="relu", name="frame5"),
        ]
    return Sequential(tdnn + [
        GlobalMeanStddevPooling1D(name="stats_pooling"),
        Dense(512, activation=None, name="segment6"),
        ReLU(name="segment6_relu"),
        Dense(512, activation="relu", name="segment7"),
        Dense(num_outputs, activation=output_activation, name="output"),
    ])

def extract_xvectors(model, inputs):
    """Generator that embeds all elements of a tf.data.Dataset as X-vectors using the given, pretrained Sequential instance returned by 'loader'."""
    xvector_extractor = Sequential([model.get_layer(l) for l in xvector_layer_names])
    for x in inputs:
        yield xvector_extractor(x)

def predict(model, inputs):
    return model.predict(inputs)
