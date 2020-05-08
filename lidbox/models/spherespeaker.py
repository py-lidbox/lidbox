"""
SphereSpeaker architecture by
Tuomas Kaseva, Aku Rouhe, Mikko Kurimo (2019).
"Spherediar: an effective speaker diarization system for meeting data".
In ASRU 2019.
https://doi.org/10.1109/ASRU46091.2019.9003967
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Layer,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def as_embedding_extractor(keras_model):
    l2_layer = keras_model.get_layer(name="l2_normalize")
    return tf.keras.models.Model(inputs=keras_model.inputs, outputs=l2_layer.output)


class L2Normalize(Layer):
    """Normalize each input to unit norm."""
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


# From https://github.com/Livefull/SphereDiar/blob/0de1683a2e333b4ebbd2ef8deacd5684449b8799/models/current_best.h5
# VLAD replaced by mean pooling
def loader(input_shape, num_outputs, embedding_dim=1000, output_activation="log_softmax"):
    inputs = Input(shape=input_shape, name="input")
    blstm_1 = Bidirectional(LSTM(250, return_sequences=True), name="blstm_1")(inputs)
    blstm_2 = Bidirectional(LSTM(250, return_sequences=True), name="blstm_2")(blstm_1)
    blstm_3 = Bidirectional(LSTM(250, return_sequences=True), name="blstm_3")(blstm_2)
    x = Concatenate(name="blstm_concat")([blstm_1, blstm_2, blstm_3])
    x = BatchNormalization(name="blstm_bn")(x)
    x = Dense(embedding_dim, activation="relu", name="fc_relu")(x)
    x = GlobalAveragePooling1D(name="avg_pooling")(x)
    x = BatchNormalization(name="pool_bn")(x)
    x = L2Normalize(name="l2_normalize")(x)
    outputs = Dense(num_outputs, name="outputs")(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="spherespeaker")
