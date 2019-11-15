"""
CNN followed by LSTMs.
Idea applied from Bartz, C. et al. (2017) "Language identification using deep convolutional recurrent neural networks".
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    LSTM,
    MaxPool2D,
    Permute,
    Reshape,
)
from tensorflow.keras import Sequential

def loader(input_shape, output_shape):
    crnn = Sequential([
        Conv2D(128, 3, input_shape=input_shape, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(128, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(256, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(512, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(512, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
    ])
    y, x, channels = crnn.layers[-1].output_shape[1:]
    timesteps = y * x
    lstm = [
        Reshape((timesteps, channels)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(output_shape, activation="softmax"),
    ]
    for layer in lstm:
        crnn.add(layer)
    return crnn
