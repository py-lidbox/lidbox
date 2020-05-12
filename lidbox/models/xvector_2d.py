"""
clstm.py with 2D CNN frontend only.
I.e. x-vector with Conv2D layers for gathering frequency channel information.
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Input,
    Layer,
    Reshape,
)
from tensorflow.keras.models import Model
import tensorflow as tf

from .xvector import (
    FrameLayer,
    GlobalMeanStddevPooling1D,
    SegmentLayer,
    as_embedding_extractor,
)


class FrameLayer2D(Layer):
    def __init__(self, filters, kernel_size, strides, name="frame", activation="relu", padding="valid", dropout_rate=None):
        super().__init__(name=name)
        self.conv = Conv2D(
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
        base_config = super().get_config()
        base_config.update({
            "filters": self.conv.filters,
            "kernel_size": self.conv.kernel_size,
            "strides": self.conv.strides,
            "padding": self.conv.padding,
            "dropout": self.dropout.rate if self.dropout else None
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def loader(input_shape, num_outputs, output_activation="log_softmax"):
    inputs = Input(shape=input_shape, name="input")
    x = Reshape((input_shape[0] or -1, input_shape[1], 1), name="reshape_to_image")(inputs)
    x = FrameLayer2D(256, (1, 5), (1, 1), name="frame2d_1")(x)
    x = FrameLayer2D(128, (1, 3), (1, 2), name="frame2d_2")(x)
    x = FrameLayer2D(64, (1, 3), (1, 3), name="frame2d_3")(x)
    x = FrameLayer2D(32, (1, 3), (1, 3), name="frame2d_4")(x)
    rows, cols, channels = x.shape[1:]
    x = Reshape((rows or -1, cols * channels), name="flatten_channels")(x)
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
    return Model(inputs=inputs, outputs=outputs, name="x-vector-2D")
