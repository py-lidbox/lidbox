"""
CNN + average pooling for variable length input utterances from
Suwon Shon, Ahmed Ali, and James Glass (June 2018). “Convolutional Neural Network and Language Embeddings for End-to-End Dialect Recognition”.
See also
https://github.com/swshon/dialectID_e2e/blob/20bec9bea05747fcf4845921498f8b3abf52c7c6/models/e2e_model.py
"""
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
)
from tensorflow.keras import Sequential

def loader(input_shape, num_outputs):
    return Sequential([
        Conv1D(500, 5, 1, input_shape=input_shape, padding="same", activation="relu", name="conv_1"),
        Conv1D(500, 7, 2, padding="same", activation="relu", name="conv_2"),
        Conv1D(500, 1, 1, padding="same", activation="relu", name="conv_3"),
        Conv1D(3000, 1, 1, padding="same", activation="relu", name="conv_4"),
        GlobalAveragePooling1D(name="avg_pooling"),
        Dense(1500, activation="relu", name="fc_1"),
        Dense(600, activation="relu", name="fc_2"),
        Dense(num_outputs, activation="softmax", name="output"),
    ])

def predict(model, samples):
    return samples
