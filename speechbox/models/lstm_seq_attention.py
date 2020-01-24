import os
os.environ.setdefault("TF_KERAS", "1")
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import (
    Activation,
    Dense,
    GlobalAveragePooling1D,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def loader(input_shape, num_outputs, output_activation="softmax", num_units=1024, attention_width=20):
    inputs = Input(shape=input_shape, name="input")
    x = LSTM(num_units, name="lstm", return_sequences=True)(inputs)
    x = SeqSelfAttention(
            units=num_units//4,
            attention_width=attention_width,
            attention_activation="sigmoid",
            name="seq_self_attention")(x)
    x = GlobalAveragePooling1D(name="time_pooling")(x)
    outputs = Dense(num_outputs, name="output", activation=None)(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, utterances):
    return model.predict(utterances)
