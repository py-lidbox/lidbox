"""
TensorFlow implementation of the x-vector extension with 2-dimensional time-frequency attention mechanism by
Miao, X., McLoughlin, I., Yan, Y. (2019) "A New Time-Frequency Attention Mechanism for TDNN and CNN-LSTM-TDNN, with Application to Language Identification".
In: Proc. Interspeech 2019, 4080-4084
DOI: 10.21437/Interspeech.2019-1256.
URL: https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1256.pdf
"""
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dropout,
    Dense,
    GaussianNoise,
    Input,
    Layer,
    LSTM,
    Multiply,
    Reshape,
)
from .xvector import (
    FrameLayer,
    GlobalMeanStddevPooling1D,
    SegmentLayer,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def frequency_attention(H, d_a=64, d_f=16):
    assert not H.shape[2] % d_f, "amount of frequency channels ({}) must be evenly divisible by the amount of frequency attention bins (d_f={})".format(H.shape[2], d_f)
    # Note, we assume that H.shape = (batch_size, T, d_h), but the paper assumes the timesteps come last
    x = Dense(d_a, activation="relu", use_bias=False, name="Wf_1")(H)
    F_A = Dense(d_f, activation="softmax", use_bias=False, name="Wf_2")(x)
    # Apply frequency attention on d_f bins
    F_A = Reshape((F_A.shape[1] or -1, F_A.shape[2], 1), name="expand_bin_weight_dim")(F_A)
    H_bins = Reshape((H.shape[1] or -1, d_f, H.shape[2] // d_f), name="partition_freq_bins")(H)
    H_bins = Multiply(name="freq_attention")([F_A, H_bins])
    # Merge weighted frequency bins
    H_weighted = Reshape((H.shape[1] or -1, H.shape[2]), name="merge_weighted_bins")(H_bins)
    return H_weighted


def loader(input_shape, num_outputs, output_activation="log_softmax", use_attention=False, use_conv2d=False, use_lstm=False):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    x = GaussianNoise(stddev=0.01, name="input_noise")(x)
    x = Dropout(rate=0.4, noise_shape=(None, 1, input_shape[1]), name="channel_dropout")(x)
    if use_conv2d:
        x = Reshape((input_shape[0] or -1, input_shape[1], 1), name="reshape_to_image")(x)
        x = Conv2D(128, (3, 9), (1, 6), activation=None, padding="same", name="conv2d_1")(x)
        x = BatchNormalization(name="conv2d_1_bn")(x)
        x = Activation("relu", name="conv2d_1_relu")(x)
        x = Conv2D(256, (3, 9), (1, 6), activation=None, padding="same", name="conv2d_2")(x)
        x = BatchNormalization(name="conv2d_2_bn")(x)
        x = Activation("relu", name="conv2d_2_relu")(x)
        # x = Reshape((x.shape[1] or -1, x.shape[2] * x.shape[3]), name="flatten_image_channels")(x)
        x = tf.math.reduce_max(x, axis=2, name="maxpool_image_channels")
    x = FrameLayer(512, 5, 1, name="frame1")(x)
    x = FrameLayer(512, 3, 2, name="frame2")(x)
    x = FrameLayer(512, 3, 3, name="frame3")(x)
    if use_lstm:
        x = LSTM(512, name="lstm", return_sequences=True)(x)
    x = FrameLayer(512, 1, 1, name="frame4")(x)
    x = FrameLayer(1500, 1, 1, name="frame5")(x)
    if use_attention:
        x = frequency_attention(x, d_f=60)
    x = GlobalMeanStddevPooling1D(name="stats_pooling")(x)
    x = SegmentLayer(512, name="segment1")(x)
    x = SegmentLayer(512, name="segment2")(x)
    outputs = Dense(num_outputs, name="output", activation=None)(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="CLSTM")
