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
from tensorflow.keras import backend


xvector_layer_names = ["frame{:d}".format(i) for i in range(1, 6)] + ["stats_pooling", "segment6"]

class StatsPooling(Layer):
    """Compute arithmetic mean and standard deviation of the inputs over the time dimension, then output the concatenation of the computed stats."""
    def call(self, inputs):
        # channels_last
        steps_axis = 1
        mean = backend.mean(inputs, axis=steps_axis)
        std = backend.std(inputs, axis=steps_axis)
        return backend.concatenate((mean, std), axis=1)


def loader(input_shape, num_outputs):
    return Sequential([
        Conv1D(512, 5, 1, padding="same", activation="relu", name="frame1", input_shape=input_shape),
        Conv1D(512, 5, 1, padding="same", activation="relu", name="frame2"),
        Conv1D(512, 7, 1, padding="same", activation="relu", name="frame3"),
        Conv1D(512, 1, 1, padding="same", activation="relu", name="frame4"),
        Conv1D(1500, 1, 1, padding="same", activation="relu", name="frame5"),
        StatsPooling(name="stats_pooling"),
        Dense(512, activation=None, name="segment6"),
        ReLU(name="segment6_relu"),
        Dense(512, activation="relu", name="segment7"),
        Dense(num_outputs, activation="softmax", name="softmax"),
    ])

def extract_xvectors(model, inputs):
    """Generator that embeds all elements of a tf.data.Dataset as X-vectors using the given, pretrained Sequential instance returned by 'loader'."""
    xvector_extractor = Sequential([model.get_layer(l) for l in xvector_layer_names])
    for x in inputs:
        yield xvector_extractor(x)

def predict(model, inputs):
    return model.predict(inputs)
