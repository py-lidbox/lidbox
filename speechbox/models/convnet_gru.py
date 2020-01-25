from tensorflow.keras.applications import ResNet50V2, MobileNetV2
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GaussianNoise,
    GlobalAveragePooling1D,
    GRU,
    Input,
    Permute,
    Reshape,
)
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


def loader(input_shape, num_outputs, num_gru_units=256, core="resnet50_v2", output_activation="softmax"):
    # Normalize and regularize by adding Gaussian noise to input (during training only)
    inputs = Input(shape=input_shape, name="input")
    x = BatchNormalization(name="input_normalization")(inputs)
    x = GaussianNoise(stddev=0.01, name="input_noise")(x)
    x = Dropout(0.2, name="input_dropout")(x)
    # Reshape into images where timesteps are columns
    images = Reshape((*input_shape, 1), name="expand_channel_dim")(x)
    images = Permute((2, 1, 3), name="freq_bins_first")(images)
    # Connect untrained ResNet50 or MobileNet architecture without inputs and outputs
    if core == "mobilenet_v2":
        convnet = MobileNetV2(include_top=False, weights=None, input_tensor=images)
    elif core == "resnet50_v2":
        convnet = ResNet50V2(include_top=False, weights=None, input_tensor=images)
    # GRU over timesteps
    x = Permute((2, 1, 3), name="timesteps_first")(convnet.layers[-1].output)
    cols, rows, channels = x.shape[1:]
    x = Reshape((cols, rows * channels), name="flatten_channels")(x)
    # x = Dense(1024, activation="sigmoid", name="sigmoid_embedding")(x)
    x = GRU(num_gru_units, name="gru", return_sequences=True)(x)
    x = BatchNormalization(name="gru_bn")(x)
    x = Dropout(0.2, name="gru_dropout")(x)
    # Pooling and output
    x = GlobalAveragePooling1D(name="timesteps_pooling")(x)
    outputs = Dense(num_outputs, activation=None, name="output")(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def predict(model, utterances):
    return np.stack([model.predict(frames).mean(axis=0) for frames in utterances.unbatch()])
