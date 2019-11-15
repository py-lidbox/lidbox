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
from tensorflow.keras import Sequential, Model
import tensorflow as tf


class ResnetBlock(Model):
    """
    Combination of
    https://adventuresinmachinelearning.com/introduction-resnet-tensorflow-2/
    and
    https://www.tensorflow.org/tutorials/customization/custom_layers
    """
    def __init__(self, num_filters, kernel_size, name=''):
        super().__init__(name=name)
        self.conv2a = Conv2D(num_filters, kernel_size, padding="same", activation="relu")
        self.bn2a = BatchNormalization()
        self.conv2b = Conv2D(num_filters, kernel_size, padding="same")
        self.bn2b = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)


def loader(input_shape, output_shape, num_lstm_units=64):
    return Sequential([
        Conv2D(64, 7, 2, input_shape=input_shape, padding="same", activation="relu"),
        MaxPool2D(2),
        ResnetBlock(64, 3),
        ResnetBlock(64, 3),
        ResnetBlock(64, 3),
        ResnetBlock(64, 3),
        ResnetBlock(64, 3),
        ResnetBlock(64, 3),
        tf.keras.layers.GlobalAveragePooling2D(),
        # Bidirectional(LSTM(num_lstm_units, return_sequences=True)),
        # Bidirectional(LSTM(num_lstm_units)),
        Dense(output_shape, activation="softmax")
     ])
