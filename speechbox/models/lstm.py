from tensorflow.keras.layers import (
    Activation,
    Dense,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def loader(input_shape, num_outputs, output_activation="softmax", num_units=128):
    output_activation_fn = getattr(tf.nn, output_activation) if output_activation is not None else None
    inputs = Input(shape=input_shape, name="input")
    lstm = LSTM(num_units, name="lstm")(inputs)
    outputs = Dense(num_outputs, name="output", activation=None)(lstm)
    outputs = Activation(output_activation_fn, name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, inputs):
    return model.predict(inputs)
