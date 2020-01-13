"""
2d CNN for spectrogram images, followed by an LSTM.
Used by Bartz, C. et al. (2017) "Language identification using deep convolutional recurrent neural networks".
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
from tensorflow.keras import Model
import tensorflow as tf


def loader(input_shape, num_outputs, output_activation="softmax"):
    assert input_shape[0] >= 32 and input_shape[1] >= 32, "too few rows and/or columns in input shape for CRNN: {}, this would lead to negative shapes after pooling".format(input_shape)
    inputs = Input(shape=input_shape, name="input")

    # CNN
    conv_1 = Conv2D(16, 7, activation="relu", padding="same", name="conv_1")(inputs)
    conv_1_bn = BatchNormalization(name="conv_1_bn")(conv_1)
    pool_conv_1 = MaxPool2D(2, name="pool_conv_1")(conv_1_bn)

    conv_2 = Conv2D(32, 5, activation="relu", padding="same", name="conv_2")(pool_conv_1)
    conv_2_bn = BatchNormalization(name="conv_2_bn")(conv_2)
    pool_conv_2 = MaxPool2D(2, name="pool_conv_2")(conv_2_bn)

    conv_3 = Conv2D(64, 3, activation="relu", padding="same", name="conv_3")(pool_conv_2)
    conv_3_bn = BatchNormalization(name="conv_3_bn")(conv_3)
    pool_conv_3 = MaxPool2D(2, name="pool_conv_3")(conv_3_bn)

    conv_4 = Conv2D(128, 3, activation="relu", padding="same", name="conv_4")(pool_conv_3)
    conv_4_bn = BatchNormalization(name="conv_4_bn")(conv_4)
    pool_conv_4 = MaxPool2D(2, name="pool_conv_4")(conv_4_bn)

    conv_5 = Conv2D(256, 3, activation="relu", padding="same", name="conv_5")(pool_conv_4)
    conv_5_bn = BatchNormalization(name="conv_5_bn")(conv_5)
    pool_conv_5 = MaxPool2D(2, name="pool_conv_5")(conv_5_bn)

    # BLSTM
    timesteps_first = Permute((2, 1, 3), name="timesteps_first")(pool_conv_5)
    cols, rows, channels = timesteps_first.shape[1:]
    flatten_channels = Reshape((cols, rows * channels), name="flatten_channels")(timesteps_first)
    blstm = Bidirectional(LSTM(256), merge_mode="concat", name="blstm")(flatten_channels)

    # Output
    outputs = Dense(num_outputs, activation=None, name="output")(blstm)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, utterances):
    return model.predict(utterances)
