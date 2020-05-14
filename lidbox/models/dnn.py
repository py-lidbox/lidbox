"""
Simple time-distributed DNN.
"""
from tensorflow.keras.layers import (
    Activation,
    Dense,
    GlobalAveragePooling1D,
    Input,
)
import tensorflow as tf


def loader(input_shape, num_outputs):
    inputs = Input(shape=input_shape, name="input")
    x = Dense(200, name="fc_1", activation="relu")(inputs)
    x = Dense(400, name="fc_2", activation="relu")(x)
    x = Dense(600, name="fc_3", activation="relu")(x)
    x = Dense(800, name="fc_4", activation="relu")(x)
    x = GlobalAveragePooling1D(name="pooling")(x)
    x = Dense(num_outputs, name="output", activation=None)(x)
    outputs = Activation(tf.nn.log_softmax, name="log_softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="DNN")
