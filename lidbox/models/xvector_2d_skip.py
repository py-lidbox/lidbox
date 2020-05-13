"""
xvector_2d with a skip connection over the 2d CNN front-end.
Channel dropout can be applied to the skipped input without affecting the input to the 2d CNN front-end.
"""
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Reshape,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model
import tensorflow as tf

from .xvector import (
    FrameLayer,
    GlobalMeanStddevPooling1D,
    SegmentLayer,
    as_embedding_extractor,
)
from .xvector_2d import FrameLayer2D


def loader(input_shape, num_outputs, output_activation="log_softmax", channel_dropout_rate=0):
    inputs = Input(shape=input_shape, name="input")
    conv2d_input = inputs
    x = Reshape((input_shape[0] or -1, input_shape[1], 1), name="reshape_to_image")(conv2d_input)
    x = FrameLayer2D(256, (1, 5), (1, 1), name="frame2d_1")(x)
    x = FrameLayer2D(128, (1, 3), (1, 2), name="frame2d_2")(x)
    x = FrameLayer2D(64, (1, 3), (1, 3), name="frame2d_3")(x)
    x = FrameLayer2D(32, (1, 3), (1, 3), name="frame2d_4")(x)
    rows, cols, channels = x.shape[1:]
    conv2d_output = Reshape((rows or -1, cols * channels), name="flatten_channels")(x)
    if channel_dropout_rate > 0:
        conv2d_input = SpatialDropout1D(channel_dropout_rate, name="channel_dropout_{:.2f}".format(channel_dropout_rate))(conv2d_input)
    x = Concatenate(name="conv2d_skip")([conv2d_input, conv2d_output])
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
    return Model(inputs=inputs, outputs=outputs, name="x-vector-2D-skip")
