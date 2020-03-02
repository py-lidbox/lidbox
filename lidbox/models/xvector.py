"""
TensorFlow implementation of the x-vector TDNN using five temporal convolutions followed by stats pooling and 3 fully connected layers proposed by
David Snyder, et al. (2018) "Spoken Language Recognition using X-vectors."
In: Proc. Odyssey 2018.
URL: http://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
import tensorflow as tf


class GlobalMeanStddevPooling1D(Layer):
    """Compute arithmetic mean and standard deviation of the inputs along the time steps dimension, then output the concatenation of the computed stats."""
    def call(self, inputs):
        # assuming always channels_last
        steps_axis = 1
        means = tf.math.reduce_mean(inputs, axis=steps_axis, keepdims=True)
        variances = tf.math.reduce_mean(tf.math.square(inputs - means), axis=steps_axis)
        means = tf.squeeze(means, steps_axis)
        stddevs = tf.math.sqrt(tf.math.maximum(0.0, variances))
        return tf.concat((means, stddevs), axis=steps_axis)


class FrameLayer(Layer):
    def __init__(self, filters, kernel_size, strides, name="frame", activation="relu", padding="valid", dropout_rate=None):
        super().__init__(name=name)
        self.conv = Conv1D(
                filters,
                kernel_size,
                strides=strides,
                activation=activation,
                padding=padding,
                name="{}_conv".format(name))
        self.batch_norm = BatchNormalization(name="{}_bn".format(name))
        self.dropout = None
        if dropout_rate:
            self.dropout = Dropout(rate=dropout_rate, name="{}_dropout".format(name))

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = {
            "filters": self.conv.filters,
            "kernel_size": self.conv.kernel_size,
            "strides": self.conv.strides,
            "padding": self.conv.padding,
            "dropout": self.dropout.rate if self.dropout else None
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SegmentLayer(Layer):
    def __init__(self, units, name="segment", activation="relu", dropout_rate=None):
        super().__init__(name=name)
        self.dense = Dense(units, activation=activation, name="{}_dense".format(name))
        self.batch_norm = BatchNormalization(name="{}_bn".format(name))
        self.dropout = None
        if dropout_rate:
            self.dropout = Dropout(rate=dropout_rate, name="{}_dropout".format(name))

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.batch_norm(x, training=training)
        if self.dropout:
            x = self.dropout(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dense.units

    def get_config(self):
        config = {
            "units": self.dense.units,
            "dropout": self.dropout.rate if self.dropout else None
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def loader(input_shape, num_outputs, output_activation="log_softmax", channel_dropout_rate=0):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    if channel_dropout_rate > 0:
        x = Dropout(rate=channel_dropout_rate, noise_shape=(None, 1, input_shape[1]), name="channel_dropout")(x)
    x = FrameLayer(512, 5, 1, name="frame1")(x)
    x = FrameLayer(512, 3, 2, name="frame2")(x)
    x = FrameLayer(512, 3, 3, name="frame3")(x)
    x = FrameLayer(512, 1, 1, name="frame4")(x)
    x = FrameLayer(1500, 1, 1, name="frame5")(x)
    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)
    x = SegmentLayer(512, name="segment1")(x)
    x = SegmentLayer(512, name="segment2")(x)
    outputs = Dense(num_outputs, name="output", activation=None)(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="x-vector")


def predict(model, inputs):
    return model.predict(inputs)
