"""
bi-GRU RNN used by
Mateju, Lukas et al. (2018). "Using Deep Neural Networks for Identification of Slavic Languages from Acoustic Signal". In: Proc. Interspeech 2018, pp. 1803–1807. DOI: 10.21437/Interspeech.2018-1165. URL: http://dx.doi.org/10.21437/Interspeech.2018-1165.
"""
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    GRU,
)
from tensorflow.keras import Sequential

def loader(input_shape, num_outputs, merge_mode="concat", num_gru_units=1024):
    return Sequential([
        Bidirectional(GRU(num_gru_units, return_sequences=True), input_shape=input_shape, merge_mode=merge_mode, name="biGRU_1"),
        Bidirectional(GRU(num_gru_units), merge_mode=merge_mode, name="biGRU_2"),
        Dense(1024, activation="relu", name="fc_relu"),
        Dense(num_outputs, activation="softmax", name="output"),
    ])

def predict(model, utterances):
    for frames in utterances.unbatch():
        yield model.predict(frames).mean(axis=0)
