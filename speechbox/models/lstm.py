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
from .xvector import (SegmentLayer, GlobalMeanStddevPooling1D)

def loader(input_shape, num_outputs, output_activation="softmax", num_units=128):
    inputs = Input(shape=input_shape, name="input")
    lstm_1 = LSTM(num_units, name="lstm_1", return_sequences=True)(inputs)
    lstm_2 = LSTM(num_units, name="lstm_2", return_sequences=True)(lstm_1)
    lstm_3 = LSTM(num_units, name="lstm_3")(lstm_2)
    outputs = Dense(num_outputs, name="output", activation=None)(lstm_3)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)

def predict(model, utterances):
    return np.stack([model.predict(frames).mean(axis=0) for frames in utterances.unbatch()])
