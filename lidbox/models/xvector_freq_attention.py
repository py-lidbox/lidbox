"""
xvector.py with frequency attention from clstm.py
"""
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Input,
)
from .xvector import (
    frame_layer,
    GlobalMeanStddevPooling1D,
    segment_layer,
    as_embedding_extractor,
)
from .clstm import frequency_attention
from tensorflow.keras.models import Model
import tensorflow as tf


def create(input_shape, num_outputs, output_activation="log_softmax", freq_attention_bins=60):
    inputs = Input(shape=input_shape, name="input")

    x = frame_layer(512, 5, 1, name="frame1")(inputs)
    x = frame_layer(512, 3, 2, name="frame2")(x)
    x = frame_layer(512, 3, 3, name="frame3")(x)
    x = frame_layer(512, 1, 1, name="frame4")(x)
    x = frame_layer(1500, 1, 1, name="frame5")(x)

    x = frequency_attention(x, d_f=freq_attention_bins)

    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)

    x = segment_layer(512, name="segment1")(x)
    x = segment_layer(512, name="segment2")(x)

    outputs = Dense(num_outputs, name="output", activation=None)(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="x-vector-frequency-attention")
