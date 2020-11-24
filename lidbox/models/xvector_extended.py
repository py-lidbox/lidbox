"""
Extended x-vector architecture by
Villalba et al. (2018) "The JHU-MIT System Description for NIST SRE18"
url: https://pdfs.semanticscholar.org/00a4/e57e9189162dc9875a1cdca527711f373b53.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
import tensorflow as tf

from .xvector import (
    frame_layer,
    GlobalMeanStddevPooling1D,
    segment_layer,
    as_embedding_extractor,
)


def create(input_shape, num_outputs, output_activation="log_softmax"):
    inputs = Input(shape=input_shape, name="input")

    x = frame_layer(512, 5, 1, name="frame1")(inputs)
    x = frame_layer(512, 1, 1, name="frame2")(x)
    x = frame_layer(512, 3, 2, name="frame3")(x)
    x = frame_layer(512, 1, 1, name="frame4")(x)
    x = frame_layer(512, 3, 3, name="frame5")(x)
    x = frame_layer(512, 1, 1, name="frame6")(x)
    x = frame_layer(512, 3, 4, name="frame7")(x)
    x = frame_layer(512, 1, 1, name="frame8")(x)
    x = frame_layer(512, 1, 1, name="frame9")(x)
    x = frame_layer(1500, 1, 1, name="frame10")(x)

    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)
    x = segment_layer(512, name="segment1")(x)
    x = segment_layer(512, name="segment2")(x)

    outputs = Dense(num_outputs, name="output", activation=None)(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="x-vector-extended")
