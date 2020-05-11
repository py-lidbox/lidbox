"""
TensorFlow implementation of the CNN + average pooling model for variable length input utterances by
Suwon Shon, Ahmed Ali, and James Glass (June 2018) "Convolutional Neural Network and Language Embeddings for End-to-End Dialect Recognition".
See also
https://github.com/swshon/dialectID_e2e/blob/20bec9bea05747fcf4845921498f8b3abf52c7c6/models/e2e_model.py
"""
from tensorflow.keras.layers import (
    Activation,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Input,
    SpatialDropout1D,
)
from tensorflow.keras import Model
import tensorflow as tf


def as_embedding_extractor(keras_model):
    fc = keras_model.get_layer(name="fc_1")
    fc.activation = None
    return tf.keras.models.Model(inputs=keras_model.inputs, outputs=fc.output)


def loader(input_shape, num_outputs, output_activation="log_softmax", padding="causal", channel_dropout_rate=0):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    if channel_dropout_rate > 0:
        x = SpatialDropout1D(channel_dropout_rate, name="channel_dropout_{:.2f}".format(channel_dropout_rate))(x)
    conv_1 = Conv1D(500, 5, 1, padding=padding, activation="relu", name="conv_1")(x)
    conv_2 = Conv1D(500, 7, 2, padding=padding, activation="relu", name="conv_2")(conv_1)
    conv_3 = Conv1D(500, 1, 1, padding=padding, activation="relu", name="conv_3")(conv_2)
    conv_4 = Conv1D(3000, 1, 1, padding=padding, activation="relu", name="conv_4")(conv_3)
    avg_pooling = GlobalAveragePooling1D(name="avg_pooling")(conv_4)
    fc_1 = Dense(1500, activation="relu", name="fc_1")(avg_pooling)
    fc_2 = Dense(600, activation="relu", name="fc_2")(fc_1)
    outputs = Dense(num_outputs, name="output", activation=None)(fc_2)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="MGB-3_CNN")
