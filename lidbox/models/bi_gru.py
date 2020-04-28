"""
TensorFlow implementation of the bi-GRU RNN used by
Mateju, Lukas et al. (2018) "Using Deep Neural Networks for Identification of Slavic Languages from Acoustic Signal".
In: Proc. Interspeech 2018, pp. 1803–1807.
URL: http://dx.doi.org/10.21437/Interspeech.2018-1165.
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Dense,
    GRU,
    Input,
)
from tensorflow.keras import Model
import tensorflow as tf


def loader(input_shape, num_outputs, output_activation="log_softmax"):
    inputs = Input(shape=input_shape, name="input")
    x = Bidirectional(GRU(512, return_sequences=True), merge_mode="concat", name="biGRU_1")(inputs)
    x = Bidirectional(GRU(512), merge_mode="concat", name="biGRU_2")(x)
    x = BatchNormalization(name="biGRU_2_bn")(x)
    x = Dense(1024, activation="relu", name="fc_relu_1")(x)
    x = BatchNormalization(name="fc_relu_1_bn")(x)
    x = Dense(1024, activation="relu", name="fc_relu_2")(x)
    x = BatchNormalization(name="fc_relu_2_bn")(x)
    outputs = Dense(num_outputs, activation=None, name="output")(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="BGRU")
