"""
Batch-normalized version of the bi-GRU RNN used by
Mateju, Lukas et al. (2018). "Using Deep Neural Networks for Identification of Slavic Languages from Acoustic Signal". In: Proc. Interspeech 2018, pp. 1803–1807. DOI: 10.21437/Interspeech.2018-1165. URL: http://dx.doi.org/10.21437/Interspeech.2018-1165.
"""
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Dense,
    GRU,
    Input,
)
from tensorflow.keras import Model
import numpy as np

def loader(input_shape, num_outputs, merge_mode="concat", num_gru_units=1024, batch_normalize=False):
    inputs = Input(shape=input_shape, name="input")
    bigru1 = Bidirectional(GRU(num_gru_units, return_sequences=True), merge_mode=merge_mode, name="biGRU_1")(inputs)
    bigru2 = Bidirectional(GRU(num_gru_units), merge_mode=merge_mode, name="biGRU_2")(bigru1)
    if batch_normalize:
        bigru2 = BatchNormalization(name="biGRU_2_bn")(bigru2)
    fc_relu1 = Dense(1024, activation="relu", name="fc_relu")(bigru2)
    if batch_normalize:
        fc_relu1 = BatchNormalization(name="fc_relu_bn")(fc_relu1)
    outputs = Dense(num_outputs, activation="softmax", name="output")(fc_relu1)
    return Model(inputs=inputs, outputs=outputs)

def predict(model, utterances):
    return np.stack([model.predict(frames).mean(axis=0) for frames in utterances.unbatch()])
