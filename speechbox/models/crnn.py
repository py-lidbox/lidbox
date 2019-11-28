"""
CNN followed by LSTMs from Bartz, C. et al. (2017) "Language identification using deep convolutional recurrent neural networks".
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    LSTM,
    MaxPool2D,
    Reshape,
)
from tensorflow.keras import Sequential

def loader(input_shape, output_shape):
    crnn = Sequential([
        Conv2D(16, 7, input_shape=input_shape, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(32, 5, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(256, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
    ])
    y, x, channels = crnn.layers[-1].output_shape[1:]
    timesteps = y * x
    rnn = [
        Reshape((timesteps, channels)),
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(256)),
    ]
    for layer in rnn:
        crnn.add(layer)
    crnn.add(Dense(output_shape, activation="softmax"))
    return crnn
