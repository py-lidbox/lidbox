"""
ResNet50V2 and MobileNetV2 wrapper for training embedding spaces.
"""
from tensorflow.keras.applications import ResNet50V2, MobileNetV2
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GaussianNoise,
    GlobalAveragePooling1D,
    Input,
    Permute,
    Reshape,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def loader(input_shape, num_outputs, core="resnet50_v2", embedding_dim=512, output_activation="log_softmax", channel_dropout_rate=0):
    inputs = Input(shape=input_shape, name="input")
    x = inputs
    if channel_dropout_rate > 0:
        x = Dropout(rate=channel_dropout_rate, noise_shape=(None, 1, input_shape[1]), name="channel_dropout")(x)
    x = Reshape((input_shape[0] or -1, input_shape[1], 1), name="reshape_to_image")(x)
    # Connect untrained Resnet50 or MobileNet architecture without inputs and outputs
    if core == "mobilenet_v2":
        convnet = MobileNetV2(include_top=False, weights=None, input_tensor=x)
    elif core == "resnet50_v2":
        convnet = ResNet50V2(include_top=False, weights=None, input_tensor=x)
    # Embedding layer with timesteps
    rows, cols, channels = convnet.output.shape[1:]
    x = Reshape((rows or -1, cols * channels), name="flatten_channels")(convnet.output)
    x = Dense(embedding_dim, activation=None, name="embedding")(x)
    x = Activation("relu", name="embedding_relu")(x)
    x = BatchNormalization(name="embedding_bn")(x)
    # Pooling and output
    x = GlobalAveragePooling1D(name="timesteps_pooling")(x)
    outputs = Dense(num_outputs, activation=None, name="output")(x)
    if output_activation:
        outputs = Activation(getattr(tf.nn, output_activation), name=str(output_activation))(outputs)
    return Model(inputs=inputs, outputs=outputs, name="{}_extractor".format(core))


def extract_embeddings(model, input_iterator):
    bnf_extractor = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
    # This requires default eager execution (TensorFlow 2) if input_iterator is a tf.data.Dataset instance.
    for x, *rest in input_iterator:
        yield (bnf_extractor.predict(x), *rest)
