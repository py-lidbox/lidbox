"""
x-vector.py with MobileNetV2 frontend for gathering frequency channel information.
"""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Input,
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


def loader(input_shape, num_outputs, output_activation="log_softmax", channel_dropout_rate=0, mean_merge_freq_channels=True):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    if channel_dropout_rate > 0:
        x = Dropout(rate=channel_dropout_rate, noise_shape=(None, 1, input_shape[1]), name="channel_dropout")(x)
    # Connect untrained MobileNetV2 CNN architecture
    x = Reshape((input_shape[0] or -1, input_shape[1], 1), name="reshape_to_image")(x)
    mobilenet = MobileNetV2(include_top=False, weights=None, input_tensor=x)
    rows, cols, channels = mobilenet.output.shape[1:]
    if mean_merge_freq_channels:
        x = tf.math.reduce_mean(mobilenet.output, axis=-2, name="mean_pooling")
    else:
        x = Reshape((rows or -1, cols * channels), name="flatten_channels")(mobilenet.output)
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
    return Model(inputs=inputs, outputs=outputs, name="mobilenet50-x-vector")
