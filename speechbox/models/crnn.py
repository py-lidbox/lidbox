"""
CNN followed by an LSTM, used in Bartz, C. et al. (2017) "Language identification using deep convolutional recurrent neural networks".
See also
https://github.com/HPI-DeepLearning/crnn-lid/blob/d78d5db14c4ee21b2cfcb09bf1d9187486371989/keras/models/crnn.py
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    LSTM,
    MaxPool2D,
    Permute,
    Reshape,
)
from tensorflow.keras import Sequential

def loader(input_shape, num_outputs):
    assert input_shape[0] >= 32 and input_shape[1] >= 32, "too few rows and/or columns in input shape for CRNN: {}, this would lead to negative shapes after pooling".format(input_shape)
    crnn = Sequential([
        Conv2D(16, 7, input_shape=input_shape, activation="relu", padding="same", name="conv_1"),
        BatchNormalization(name="bn_conv_1"),
        MaxPool2D(2, name="pool_conv_1"),
        Conv2D(32, 5, activation="relu", padding="same", name="conv_2"),
        BatchNormalization(name="bn_conv_2"),
        MaxPool2D(2, name="pool_conv_2"),
        Conv2D(64, 3, activation="relu", padding="same", name="conv_3"),
        BatchNormalization(name="bn_conv_3"),
        MaxPool2D(2, name="pool_conv_3"),
        Conv2D(128, 3, activation="relu", padding="same", name="conv_4"),
        BatchNormalization(name="bn_conv_4"),
        MaxPool2D(2, name="pool_conv_4"),
        Conv2D(256, 3, activation="relu", padding="same", name="conv_5"),
        BatchNormalization(name="bn_conv_5"),
        MaxPool2D(2, name="pool_conv_5"),
    ])
    crnn.add(Permute((2, 1, 3), name="timesteps_first"))
    cols, rows, channels = crnn.layers[-1].output_shape[1:]
    crnn.add(Reshape((cols, rows * channels), name="merge_channels"))
    crnn.add(Bidirectional(LSTM(256), merge_mode="concat", name="blstm"))
    crnn.add(Dense(num_outputs, activation="softmax", name="output"))
    return crnn

def predict(*args):
    pass
