"""
TensorFlow implementation of the X-vector DNN consisting of five temporal convolutions followed by stats pooling and 3 fully connected layers.
See also:
David Snyder, et al. "Spoken Language Recognition using X-vectors." Odyssey. 2018.
http://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
"""
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Input,
    Layer,
    ReLU,
)
from tensorflow.keras.models import Sequential
import tensorflow as tf

xvector_layer_names = ["frame{:d}".format(i) for i in range(1, 6)] + ["stats_pooling", "segment6"]

class GlobalMeanStddevPooling1D(Layer):
    """Compute arithmetic mean and standard deviation of the inputs over the time dimension, then output the concatenation of the computed stats."""
    def call(self, inputs):
        # channels_last
        steps_axis = 1
        mean = tf.math.reduce_mean(inputs, axis=steps_axis)
        std = tf.math.reduce_std(inputs, axis=steps_axis)
        return tf.concat((mean, std), axis=1)

def loader(input_shape, num_outputs, output_activation="softmax"):
    return Sequential([
        Conv1D( 512, 5, 1, name="frame1", padding="valid", activation="relu", input_shape=input_shape),
        Conv1D( 512, 3, 2, name="frame2", padding="valid", activation="relu"),
        Conv1D( 512, 3, 3, name="frame3", padding="valid", activation="relu"),
        Conv1D( 512, 1, 1, name="frame4", padding="valid", activation="relu"),
        Conv1D(1500, 1, 1, name="frame5", padding="valid", activation="relu"),
        GlobalMeanStddevPooling1D(name="stats_pooling"),
        Dense(512, name="segment6", activation=None),
        ReLU(name="segment6_relu"),
        Dense(512, name="segment7", activation="relu"),
        Dense(num_outputs, name="output", activation=output_activation),
    ])

def extract_xvectors(model, inputs):
    """Generator that embeds all elements of a tf.data.Dataset as X-vectors using the given, pretrained Sequential instance returned by 'loader'."""
    xvector_extractor = Sequential([model.get_layer(l) for l in xvector_layer_names])
    for x in inputs:
        yield xvector_extractor(x)

def predict(model, inputs):
    return model.predict(inputs)
