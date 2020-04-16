"""
TensorFlow implementation of the BLSTM-based language vector extractor used by
Gelly, G. and J.L. Gauvain (Aug. 2017) "Spoken Language Identification Using LSTM-Based Angular Proximity".
In: Proc. Interspeech 2017
URL: https://www.isca-speech.org/archive/Interspeech_2017/abstracts/1334.html
"""
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    GlobalAveragePooling1D,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def loader(input_shape, num_outputs):
    inputs = Input(shape=input_shape, name="input")
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True), name="blstm_1")(inputs)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True), name="blstm_2")(blstm_1)
    concat = Concatenate(name="blstm_concat")([blstm_1, blstm_2])
    # TODO L2 normalization
    outputs = GlobalAveragePooling1D(name="avg_over_time")(concat)
    return Model(inputs=inputs, outputs=outputs, name="angular_proximity_lstm")
