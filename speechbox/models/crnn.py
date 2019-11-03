"""
CNN followed by an LSTM layer.
Idea applied from Bartz, C. et al. (2017) "Language identification using deep convolutional recurrent neural networks".
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
from tensorflow.keras.models import Sequential

def loader(input_shape, output_shape, filters=(32, 64, 128, 256), lstm_units=128, blstm=False):
    m = Sequential()
    for i, num_filters in enumerate(filters):
        if i:
            m.add(Conv2D(num_filters, 3, activation="relu"))
        else:
            m.add(Conv2D(num_filters, 3, activation="relu", input_shape=input_shape))
        m.add(BatchNormalization())
        m.add(MaxPool2D())
    m.add(Permute((2, 1, 3)))
    y, x, timesteps = m.layers[-1].output_shape[1:]
    m.add(Reshape((y, x * timesteps)))
    lstm = LSTM(lstm_units)
    if blstm:
        lstm = Bidirectional(lstm)
    m.add(lstm)
    m.add(Dense(output_shape, activation="softmax"))
    return m
