"""
X-Vector TDNN using five temporal convolutions followed by stats pooling and 3 fully connected layers proposed by:
David Snyder, et al. (2018) "Spoken Language Recognition using X-vectors."
In: Proc. Odyssey 2018.
URL: http://danielpovey.com/files/2018_odyssey_xvector_lid.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Layer,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model
import tensorflow as tf

# Assuming spectral features (Batch, Time, Channels), where freq. channels are always last
TIME_AXIS = 1
STDDEV_SQRT_MIN_CLIP = 1e-10


class GlobalMeanStddevPooling1D(Layer):
    """
    Compute arithmetic mean and standard deviation of the inputs along the time steps dimension,
    then output the concatenation of the computed stats.
    """
    def call(self, inputs):
        means = tf.math.reduce_mean(inputs, axis=TIME_AXIS, keepdims=True)
        variances = tf.math.reduce_mean(tf.math.square(inputs - means), axis=TIME_AXIS)
        means = tf.squeeze(means, TIME_AXIS)
        stddevs = tf.math.sqrt(tf.clip_by_value(variances, STDDEV_SQRT_MIN_CLIP, variances.dtype.max))
        return tf.concat((means, stddevs), axis=TIME_AXIS)


def create(input_shape, num_outputs, channel_dropout_rate=0, name="x-vector"):
    inputs = Input(shape=input_shape, name="input")

    x = inputs
    if channel_dropout_rate > 0:
        x = SpatialDropout1D(channel_dropout_rate, name="channel_dropout")(x)

    conv_conf = dict(padding="causal", activation="relu")
    x = Conv1D(512,  5, 1, **conv_conf, name="frame1")(x)
    x = Conv1D(512,  3, 2, **conv_conf, name="frame2")(x)
    x = Conv1D(512,  3, 3, **conv_conf, name="frame3")(x)
    x = Conv1D(512,  1, 1, **conv_conf, name="frame4")(x)
    x = Conv1D(1500, 1, 1, **conv_conf, name="frame5")(x)

    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)

    x = Dense(512, activation="relu", name="segment1")(x)
    x = Dense(512, activation="relu", name="segment2")(x)

    x = Dense(num_outputs, activation=None, name="outputs")(x)
    outputs = Activation(tf.nn.log_softmax, name="log_softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


def as_embedding_extractor(m):
    segment_layer = m.get_layer(name="segment1")
    segment_layer.activation = None
    return Model(inputs=m.inputs, outputs=segment_layer.output)
