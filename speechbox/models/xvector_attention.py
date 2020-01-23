"""
X-vector architecture with multi-level attention after global pooling.
Attention from
https://www.researchgate.net/profile/Karim_Said_Barsim/publication/323627323_Multi-level_Attention_Model_for_Weakly_Supervised_Audio_Classification/links/5aa294b30f7e9badd9a662dc/Multi-level-Attention-Model-for-Weakly-Supervised-Audio-Classification.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    Add,
    Concatenate,
    Dense,
    Input,
    Layer,
)
from tensorflow.keras.models import Model
import tensorflow as tf

from .xvector import (
    FrameLayer,
    GlobalMeanStddevPooling1D,
    SegmentLayer,
)


class SimpleAttention(Layer):
    def call(self, inputs):
        query = tf.nn.softmax(inputs)
        query = tf.clip_by_value(query, 1e-7, 1.0 - 1e-7)
        value = tf.nn.sigmoid(inputs)
        return query * value


def loader(input_shape, num_outputs, output_activation="log_softmax", L=2, use_attention=True, dropout_rate=None):
    inputs = Input(shape=input_shape, name="input")
    frame1 = FrameLayer(512, 5, 1, name="frame1", dropout_rate=dropout_rate)(inputs)
    frame2 = FrameLayer(512, 3, 2, name="frame2", dropout_rate=dropout_rate)(frame1)
    frame3 = FrameLayer(512, 3, 3, name="frame3", dropout_rate=dropout_rate)(frame2)
    frame4 = FrameLayer(512, 1, 1, name="frame4")(frame3)
    frame5 = FrameLayer(1500, 1, 1, name="frame5")(frame4)
    stats_pooling = GlobalMeanStddevPooling1D(name="stats_pooling")(frame5)
    x = SegmentLayer(512, name="segment1")(stats_pooling)
    if use_attention:
        attention_outputs = []
        for level in range(1, L + 1):
            att_name = "attention{}".format(level)
            x = SegmentLayer(512, name=att_name + "_segment")(x)
            att_input = SegmentLayer(num_outputs, name=att_name + "_input")(x)
            att_output = SimpleAttention(name=att_name)(att_input)
            attention_outputs.append(att_output)
        concat = Concatenate(name="attention_concat")(attention_outputs)
        outputs = Dense(num_outputs, name="outputs")(concat)
    else:
        for level in range(2, L + 2):
            x = SegmentLayer(512, name="segment{}".format(level))(x)
        outputs = Dense(num_outputs, name="outputs")(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, inputs):
    return model.predict(inputs)
