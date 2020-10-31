"""
TensorFlow JavaScript compatible implementation of xvector.py
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Layer,
    ReLU,
)
from tensorflow.keras.models import Model
import tensorflow as tf

from .xvector import (
    # Implemented separately in xvector_stats_pooling.js
    GlobalMeanStddevPooling1D,
)


def as_embedding_extractor(keras_model):
    segment_layer = keras_model.get_layer(name="segment1_dense")
    return tf.keras.models.Model(inputs=keras_model.inputs, outputs=segment_layer.output)


def FrameLayer(inputs, filters, kernel_size, stride, name="frame", activation="relu", dropout_rate=None):
    """Batch normalized temporal convolution"""
    x = Conv1D(filters, kernel_size, stride, name="{}_conv".format(name), activation=None, padding="same")(inputs)
    x = BatchNormalization(name="{}_bn".format(name))(x)
    x = Activation(activation, name="{}_{}".format(name, str(activation)))(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate, name="{}_dropout".format(name))(x)
    return x

def SegmentLayer(inputs, units, name="segment", activation="relu", dropout_rate=None):
    """Batch normalized dense layer"""
    x = Dense(units, name="{}_dense".format(name), activation=None)(inputs)
    x = BatchNormalization(name="{}_bn".format(name))(x)
    x = Activation(activation, name="{}_{}".format(name, str(activation)))(x)
    if dropout_rate:
        x = Dropout(rate=dropout_rate, name="{}_dropout".format(name))(x)
    return x

def loader(input_shape, num_outputs, output_activation="log_softmax", dropout_rate=None):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    x = FrameLayer(x, 512, 5, 1, name="frame1", dropout_rate=dropout_rate)
    x = FrameLayer(x, 512, 3, 2, name="frame2", dropout_rate=dropout_rate)
    x = FrameLayer(x, 512, 3, 3, name="frame3", dropout_rate=dropout_rate)
    x = FrameLayer(x, 512, 1, 1, name="frame4")
    x = FrameLayer(x, 1500, 1, 1, name="frame5")
    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)
    x = SegmentLayer(x, 512, name="segment1")
    x = SegmentLayer(x, 512, name="segment2")
    x = Dense(num_outputs, name="output", activation=None)(x)
    outputs = x
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="x-vector-javascript")
