"""
TensorFlow implementation of the X-vector DNN consisting of five temporal convolutions followed by stats pooling and 3 fully connected layers.
See also:
David Snyder, et al. "Spoken Language Recognition using X-vectors." Odyssey. 2018.
http://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Layer,
    ReLU,
)
from tensorflow.keras.models import Model
import tensorflow as tf


class GlobalMeanStddevPooling1D(Layer):
    """Compute arithmetic mean and standard deviation of the inputs over the time dimension, then output the concatenation of the computed stats."""
    def call(self, inputs):
        # channels_last
        steps_axis = 1
        mean = tf.math.reduce_mean(inputs, axis=steps_axis)
        std = tf.math.reduce_std(inputs, axis=steps_axis)
        return tf.concat((mean, std), axis=1)


class FrameLayer(Layer):
    def __init__(self, filters, kernel_size, stride, name="frame", activation="relu", padding="causal", dropout_rate=None):
        super().__init__(name=name)
        self.conv = Conv1D(filters, kernel_size, stride, name="{}_conv".format(name), activation=None, padding=padding)
        self.batch_norm = BatchNormalization(name="{}_bn".format(name))
        self.activation = Activation(activation, name="{}_{}".format(name, str(activation)))
        self.dropout = None
        if dropout_rate:
            self.dropout = Dropout(rate=dropout_rate, name="{}_dropout".format(name))

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def get_config(self):
        return {
            "filters": self.conv.filters,
            "kernel_size": self.conv.kernel_size[0],
            "stride": self.conv.strides[0],
            "name": super().get_config()["name"],
            "activation": self.activation.activation,
            "padding": self.conv.padding,
            "dropout": self.dropout.rate if self.dropout else None
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SegmentLayer(Layer):
    def __init__(self, units, name="segment", activation="relu", dropout_rate=None):
        super().__init__(name=name)
        self.dense = Dense(units, name="{}_dense".format(name), activation=None)
        self.batch_norm = BatchNormalization(name="{}_bn".format(name))
        self.activation = Activation(activation, name="{}_{}".format(name, str(activation)))
        self.dropout = None
        if dropout_rate:
            self.dropout = Dropout(rate=dropout_rate, name="{}_dropout".format(name))

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def get_config(self):
        return {
            "units": self.dense.units,
            "name": super().get_config()["name"],
            "activation": self.activation.activation,
            "dropout": self.dropout.rate if self.dropout else None
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def loader(input_shape, num_outputs, output_activation="softmax", dropout_rate=None):
    inputs = Input(shape=input_shape, name="input")
    frame1 = FrameLayer(512, 5, 1, name="frame1", dropout_rate=dropout_rate)(inputs)
    frame2 = FrameLayer(512, 3, 2, name="frame2", dropout_rate=dropout_rate)(frame1)
    frame3 = FrameLayer(512, 3, 3, name="frame3", dropout_rate=dropout_rate)(frame2)
    frame4 = FrameLayer(512, 1, 1, name="frame4")(frame3)
    frame5 = FrameLayer(1500, 1, 1, name="frame5")(frame4)
    stats_pooling = GlobalMeanStddevPooling1D(name="stats_pooling")(frame5)
    segment1 = SegmentLayer(512, name="segment1")(stats_pooling)
    segment2 = SegmentLayer(512, name="segment2")(segment1)
    outputs = Dense(num_outputs, name="output", activation=None)(segment2)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, inputs):
    return model.predict(inputs)
