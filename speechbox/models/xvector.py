"""
TensorFlow implementation of the X-vector DNN consisting of five temporal convolutions followed by stats pooling and 3 fully connected layers.
See also:
David Snyder, et al. "Spoken Language Recognition using X-vectors." Odyssey. 2018.
http://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
"""
from tensorflow.keras.layers import (
    BatchNormalization,
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

def loader(input_shape, num_outputs, output_activation="softmax", batch_normalization=False):
    if batch_normalization:
        model = Sequential([
            Conv1D(512, 5, 1, name="frame1", padding="valid", activation=None, input_shape=input_shape),
            BatchNormalization(name="frame1_bn"),
            ReLU(name="frame1_relu"),
            Conv1D(512, 3, 2, name="frame2", padding="valid", activation=None),
            BatchNormalization(name="frame2_bn"),
            ReLU(name="frame2_relu"),
            Conv1D(512, 3, 3, name="frame3", padding="valid", activation=None),
            BatchNormalization(name="frame3_bn"),
            ReLU(name="frame3_relu"),
            Conv1D(512, 1, 1, name="frame4", padding="valid", activation=None),
            BatchNormalization(name="frame4_bn"),
            ReLU(name="frame4_relu"),
            Conv1D(1500, 1, 1, name="frame5", padding="valid", activation=None),
            BatchNormalization(name="frame5_bn"),
            ReLU(name="frame5_relu"),
            GlobalMeanStddevPooling1D(name="stats_pooling"),
            Dense(512, name="segment6", activation=None),
            BatchNormalization(name="segment6_bn"),
            ReLU(name="segment6_relu"),
            Dense(512, name="segment7", activation=None),
            BatchNormalization(name="segment7_bn"),
            ReLU(name="segment7_relu"),
        ])
    else:
        model = Sequential([
            Conv1D(512, 5, 1, name="frame1", padding="valid", activation="relu", input_shape=input_shape),
            Conv1D(512, 3, 2, name="frame2", padding="valid", activation="relu"),
            Conv1D(512, 3, 3, name="frame3", padding="valid", activation="relu"),
            Conv1D(512, 1, 1, name="frame4", padding="valid", activation="relu"),
            Conv1D(1500, 1, 1, name="frame5", padding="valid", activation="relu"),
            GlobalMeanStddevPooling1D(name="stats_pooling"),
            Dense(512, name="segment6", activation=None),
            ReLU(name="segment6_relu"),
            Dense(512, name="segment7", activation="relu"),
        ])
    model.add(Dense(num_outputs, name="output", activation=output_activation))
    return model

def extract_xvectors(model, inputs):
    """Generator that embeds all elements of a tf.data.Dataset as X-vectors using the given, pretrained Sequential instance returned by 'loader'."""
    xvector_extractor = Sequential([model.get_layer(l) for l in xvector_layer_names])
    for x in inputs:
        yield xvector_extractor(x)

def predict(model, inputs):
    return model.predict(inputs)
