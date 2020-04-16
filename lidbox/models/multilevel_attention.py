"""
TensorFlow implementation of the multi-level attention model for classifying audio class embeddings by
Yu, Changsong et al. (2018). “Multi-level attention model for weakly supervised audio classification”.
In: DCASE2018 Workshop on Detection and Classification of Acoustic Scenes and Events.
URL: http://epubs.surrey.ac.uk/849626/.
See also https://github.com/ChangsongYu/Eusipco2018_Google_AudioSet.git
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
import tensorflow as tf


class Attention(Layer):
    def __init__(self, num_units, name="attention"):
        super().__init__(name=name)
        self.fc = Dense(num_units, name=name + "_input")

    def call(self, inputs):
        x = self.fc(inputs)
        query = tf.nn.softmax(x)
        query = tf.clip_by_value(query, 1e-7, 1.0 - 1e-7)
        query /= tf.math.reduce_sum(query, axis=1, keepdims=True)
        value = tf.nn.sigmoid(x)
        attention = tf.math.reduce_sum(query * value, axis=1)
        return attention

    def get_config(self):
        config = super().get_config()
        config.update({"num_units": self.fc.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseBlock(Layer):
    def __init__(self, H, dropout_rate, name="dense_block"):
        super().__init__(name=name)
        self.fc = Dense(H, name=name + "_fc")
        self.bn = BatchNormalization(name=name + "_bn")
        self.relu = Activation("relu", name=name + "_relu")
        self.dropout = Dropout(dropout_rate, name=name + "_dropout")

    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"H": self.fc.units, "dropout_rate": self.dropout.rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def loader(input_shape, num_outputs, output_activation="log_softmax", L=2, H=512):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    attention_outputs = []
    for level in range(1, L + 1):
        # Each DenseBlock here corresponds to "embedded mapping" in Yu's paper
        x = DenseBlock(H, 0.4, name="dense_block{}".format(level))(x)
        # Insert attention module for this level
        att_output = Attention(num_outputs, name="attention{}".format(level))(x)
        attention_outputs.append(att_output)
    concat = Concatenate(name="attention_concat")(attention_outputs)
    outputs = Dense(num_outputs, name="outputs")(concat)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="DNN_multilevel_attention")
