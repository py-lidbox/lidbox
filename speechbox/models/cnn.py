"""
CNN + average pooling for variable length input utterances from
Suwon Shon, Ahmed Ali, and James Glass (June 2018). “Convolutional Neural Network and Language Embeddings for End-to-End Dialect Recognition”.
See also
https://github.com/swshon/dialectID_e2e/blob/20bec9bea05747fcf4845921498f8b3abf52c7c6/models/e2e_model.py
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    ReLU,
)
from tensorflow.keras import Sequential

def loader(input_shape, num_outputs, variable_input_length=False):
    if variable_input_length:
        input_shape = (None, *input_shape[1:])
    return Sequential([
        Conv1D(500, 5, 1, input_shape=input_shape, padding="same", activation=None, name="conv_1"),
        BatchNormalization(name="BN_conv_1"),
        ReLU(name="conv_1_relu"),
        Conv1D(500, 7, 2, padding="same", activation=None, name="conv_2"),
        BatchNormalization(name="BN_conv_2"),
        ReLU(name="conv_2_relu"),
        Conv1D(500, 1, 1, padding="same", activation=None, name="conv_3"),
        BatchNormalization(name="BN_conv_3"),
        ReLU(name="conv_3_relu"),
        Conv1D(3000, 1, 1, padding="same", activation=None, name="conv_4"),
        BatchNormalization(name="BN_conv_4"),
        ReLU(name="conv_4_relu"),
        GlobalAveragePooling1D(name="conv_4_pooling"),
        Dense(1500, activation=None, name="dense_1"),
        BatchNormalization(name="BN_dense_1"),
        ReLU(name="dense_1_relu"),
        Dense(600, activation=None, name="dense_2"),
        BatchNormalization(name="BN_dense_2"),
        ReLU(name="dense_2_relu"),
        Dense(num_outputs, activation=None, name="output"),
    ])
