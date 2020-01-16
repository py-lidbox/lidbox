from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Dense,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

def loader(input_shape, num_outputs, output_activation="softmax", num_units=512):
    inputs = Input(shape=input_shape, name="input")
    x = Bidirectional(LSTM(num_units, name="blstm"))(inputs)
    x = Dense(2 * num_units, name="fc_relu_1", activation="relu")(x)
    x = BatchNormalization(name="fc_relu_bn_1")(x)
    x = Dense(num_units, name="fc_relu_2", activation="relu")(x)
    x = BatchNormalization(name="fc_relu_bn_2")(x)
    x = Dense(num_outputs, name="output", activation=None)(x)
    if output_activation:
        x = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(x)
    return Model(inputs=inputs, outputs=x)

def predict(model, utterances):
    return np.stack([model.predict(frames).mean(axis=0) for frames in utterances.unbatch()])
