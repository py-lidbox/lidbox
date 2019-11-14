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

def loader(input_shape, output_shape, num_lstm_units=256):
    return Sequential([
         Conv2D(64, 7, 2, input_shape=input_shape, padding="same"),
         MaxPool2D(2),
         Conv2D(64, 3, 2, padding="same"),
         BatchNormalization(),
         Conv2D(128, 3, 2, padding="same"),
         BatchNormalization(),
         Conv2D(256, 3, 2, padding="same"),
         BatchNormalization(),
         Conv2D(512, 3, 2, padding="same"),
         BatchNormalization(),
         Permute((3, 1, 2)),
         Reshape((-1, 1)),
         Bidirectional(LSTM(num_lstm_units, return_sequences=True)),
         Bidirectional(LSTM(num_lstm_units)),
         Dense(output_shape, activation="softmax")
     ])
