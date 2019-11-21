"""
Smaller sized version of the LSTM model in Wan et al (2019)
https://ieeexplore.ieee.org/abstract/document/8683313
"""
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    LSTM,
    LeakyReLU,
)
from tensorflow.keras import Sequential

def loader(input_shape, num_outputs, d=1):
    return Sequential([
        LSTM(512//d, input_shape=input_shape, return_sequences=True, name="lstm_1"),
        Conv1D(256//d, 1, name="conv_1"),
        LSTM(256//d, name="lstm_2"),
        LeakyReLU(),
        Dense(num_outputs, activation="softmax", name="output"),
    ])
