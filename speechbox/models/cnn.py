"""
Variable sized Keras adaptation of the CNN from
https://github.com/swshon/dialectID_e2e/blob/20bec9bea05747fcf4845921498f8b3abf52c7c6/models/e2e_model.py
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
)
from tensorflow.keras import Sequential

def loader(input_shape, output_shape, d=1):
    return Sequential([
        Conv1D(500//d, 5, 1, input_shape=input_shape, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(500//d, 7, 2, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(500//d, 1, 1, padding="same", activation="relu"),
        BatchNormalization(),
        Conv1D(3000//d, 1, 1, padding="same", activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(1500//d, activation="relu"),
        BatchNormalization(),
        Dense(600//d, activation="relu"),
        BatchNormalization(),
        Dense(output_shape, activation="softmax"),
    ])
