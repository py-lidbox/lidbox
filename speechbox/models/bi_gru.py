"""
bi-GRU architecture from
Mateju, Lukas et al. (2018). "Using Deep Neural Networks for Identification of Slavic Languages from Acoustic Signal". In: Proc. Interspeech 2018, pp. 1803–1807. DOI: 10.21437/Interspeech.2018-1165. URL: http://dx.doi.org/10.21437/Interspeech.2018-1165.
"""
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    GRU,
)
from tensorflow.keras import Sequential

def loader(input_shape, num_outputs, merge_mode="ave", size_reduction=1):
    x = size_reduction
    return Sequential([
        Bidirectional(GRU(1024//x, return_sequences=True), input_shape=input_shape, merge_mode=merge_mode, name="biGRU_1"),
        Bidirectional(GRU(1024//x), merge_mode=merge_mode, name="biGRU_2"),
        Dense(1024//x, activation="relu", name="relu"),
        Dense(num_outputs, activation="softmax", name="output"),
    ])
