"""
TensorFlow implementation of the BLSTM-based language vector extractor used by
Gelly, G. and J.L. Gauvain (Aug. 2017) "Spoken Language Identification Using LSTM-Based Angular Proximity".
In: Proc. Interspeech 2017
URL: https://www.isca-speech.org/archive/Interspeech_2017/abstracts/1334.html

Based on Figure 1 (right side) in the paper.
This implementation uses standard LSTM cells instead of Coordinated-Gate LSTMs.
"""
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Multiply,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def create(input_shape, num_outputs, num_lstm_units=62, alpha1=0.5, alpha2=0.5):
    inputs = Input(shape=input_shape, name="input")

    lstm_1 = LSTM(num_lstm_units, return_sequences=True, name="lstm_1")
    blstm_1 = Bidirectional(lstm_1, merge_mode="concat", name="blstm_1")(inputs)
    lstm_2 = LSTM(num_lstm_units, return_sequences=True, name="lstm_2")
    blstm_2 = Bidirectional(lstm_2, merge_mode="concat", name="blstm_2")(blstm_1)

    alpha1 = tf.constant(alpha1, dtype=blstm_1.dtype, shape=[1])
    alpha2 = tf.constant(alpha2, dtype=blstm_2.dtype, shape=[1])
    blstm_1_weighted = Multiply(name="alpha1")([alpha1, blstm_1])
    blstm_2_weighted = Multiply(name="alpha2")([alpha2, blstm_2])

    concat = Concatenate(name="blstm_concat")([blstm_1_weighted, blstm_2_weighted])
    avg_over_time = GlobalAveragePooling1D(name="avg_over_time")(concat)
    lang_vec_z = tf.math.l2_normalize(avg_over_time, axis=1)

    return Model(inputs=inputs, outputs=lang_vec_z, name="angular_proximity_lstm")
