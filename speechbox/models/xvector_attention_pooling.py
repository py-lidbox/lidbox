"""
X-vector with global pooling replaced by attention
https://www.researchgate.net/profile/Karim_Said_Barsim/publication/323627323_Multi-level_Attention_Model_for_Weakly_Supervised_Audio_Classification/links/5aa294b30f7e9badd9a662dc/Multi-level-Attention-Model-for-Weakly-Supervised-Audio-Classification.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Input,
    Layer,
    TimeDistributed,
)
from tensorflow.keras.models import Model
import tensorflow as tf

from .xvector import FrameLayer, SegmentLayer

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
        return dict(super().get_config(), num_units=self.fc.units)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def loader(input_shape, num_outputs, output_activation="softmax", L=2, dropout_rate=None):
    inputs = Input(shape=input_shape, name="input")
    frame1 = FrameLayer(512, 5, 1, name="frame1", dropout_rate=dropout_rate)(inputs)
    frame2 = FrameLayer(512, 3, 2, name="frame2", dropout_rate=dropout_rate)(frame1)
    frame3 = FrameLayer(512, 3, 3, name="frame3", dropout_rate=dropout_rate)(frame2)
    frame4 = FrameLayer(512, 1, 1, name="frame4")(frame3)
    frame5 = FrameLayer(1500, 1, 1, name="frame5")(frame4)
    x = frame5
    attention_outputs = []
    for level in range(1, L + 1):
        att_name = "attention{}".format(level)
        # This corresponds to the "embedded mapping" in Yu et al. (2018)
        x = SegmentLayer(512, name="segment{}".format(level))(x)
        # Insert attention module for this level
        att_output = Attention(num_outputs, name=att_name)(x)
        attention_outputs.append(att_output)
    concat = Concatenate(name="attention_concat")(attention_outputs)
    outputs = Dense(num_outputs, name="outputs")(concat)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, inputs):
    return model.predict(inputs)
