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
    SpatialDropout1D,
)
from tensorflow.keras.models import Model
import tensorflow as tf


# Assuming spectral features (Batch, Time, Channels), where freq. channels are always last
TIME_AXIS = 1


class GlobalMeanStddevPooling1D(Layer):
    """Compute arithmetic mean and standard deviation of the inputs along the time steps dimension, then output the concatenation of the computed stats."""
    def call(self, inputs):
        means = tf.math.reduce_mean(inputs, axis=TIME_AXIS, keepdims=True)
        variances = tf.math.reduce_mean(tf.math.square(inputs - means), axis=TIME_AXIS)
        means = tf.squeeze(means, TIME_AXIS)
        stddevs = tf.math.sqrt(tf.clip_by_value(variances, 0, variances.dtype.max))
        return tf.concat((means, stddevs), axis=TIME_AXIS)


class FrameLayer(Layer):
    def __init__(self, filters, kernel_size, strides, name="frame", activation="relu", padding="causal", dropout_rate=None):
        super().__init__(name=name)
        self.conv = Conv1D(
                filters,
                kernel_size,
                strides=strides,
                activation=activation,
                padding=padding,
                name="{}_conv".format(name))
        self.batch_norm = BatchNormalization(axis=TIME_AXIS, name="{}_bn".format(name))
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
        config = super().get_config()
        config.update({
            "filters": self.conv.filters,
            "kernel_size": self.conv.kernel_size,
            "strides": self.conv.strides,
            "padding": self.conv.padding,
            "dropout": self.dropout.rate if self.dropout else None
        })
        return config

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
        config = super().get_config()
        config.update({
            "units": self.dense.units,
            "dropout": self.dropout.rate if self.dropout else None
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def as_embedding_extractor(keras_model):
    segment_layer = keras_model.get_layer(name="segment1")
    segment_layer.dense.activation = None
    return tf.keras.models.Model(inputs=keras_model.inputs, outputs=segment_layer.dense.output)


def loader(input_shape, num_outputs, output_activation="log_softmax", channel_dropout_rate=0):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    if channel_dropout_rate > 0:
        x = SpatialDropout1D(channel_dropout_rate, name="channel_dropout_{:.2f}".format(channel_dropout_rate))(x)
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
