"""
TensorFlow implementation of the CRNN model used by
Bartz, C. et al. (2017) "Language identification using deep convolutional recurrent neural networks".
See also
https://github.com/HPI-DeepLearning/crnn-lid/blob/d78d5db14c4ee21b2cfcb09bf1d9187486371989/keras/models/crnn.py
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Input,
    LSTM,
    MaxPool2D,
    Permute,
    Reshape,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
import tensorflow as tf


def loader(input_shape, num_outputs, output_activation="softmax", weight_decay=0.001):
    inputs = Input(shape=input_shape, name="input")
    images = Reshape((*input_shape, 1), name="expand_channel_dim")(inputs)
    images = Permute((2, 1, 3), name="freq_bins_first")(images)

    # CNN
    filter_def = (16, 32, 64, 128, 256)
    kernel_def = (7, 5, 3, 3, 3)
    x = images
    for i, (f, k) in enumerate(zip(filter_def, kernel_def), start=1):
        x = Conv2D(f, k,
                activation="relu",
                kernel_regularizer=l2(weight_decay),
                padding="same",
                name="conv_{}".format(i))(x)
        x = BatchNormalization(name="conv_{}_bn".format(i))(x)
        x = MaxPool2D(2, name="conv_{}_pool".format(i))(x)

    # BLSTM
    timesteps_first = Permute((2, 1, 3), name="timesteps_first")(x)
    cols, rows, channels = timesteps_first.shape[1:]
    flatten_channels = Reshape((cols, rows * channels), name="flatten_channels")(timesteps_first)
    blstm = Bidirectional(LSTM(256), merge_mode="concat", name="blstm")(flatten_channels)

    # Output
    outputs = Dense(num_outputs, activation=None, name="output")(blstm)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="CRNN")
