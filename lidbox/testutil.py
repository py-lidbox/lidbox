"""
Utilities for testing purposes, mostly random data generation.
"""
import numpy as np
import librosa
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays


def peak_normalize(signal, dBFS):
    # https://www.hackaudio.com/digital-signal-processing/amplitude/peak-normalization/
    level = 10.0 ** (dBFS/20.0)
    return level * (signal / np.max(np.abs(signal)))


def chirps(freqs, sample_rate, duration):
    signals = []
    for f1, f2 in zip(freqs, freqs[1:]):
        signals.append(librosa.chirp(f1, f2, sr=sample_rate, duration=duration/len(freqs)))
    return np.concatenate(signals)


def noisy_sinewave(freq, sample_rate, duration):
    signal = librosa.tone(freq, sr=sample_rate, duration=duration)
    noise = np.random.normal(0, 0.1, sample_rate * duration) ** 2
    return peak_normalize(signal + noise, -3)


@composite
def spectrograms(draw, min_shape=(1, 1, 1), max_shape=(64, 1000, 128)):
    shape = []
    for dim_min, dim_max in zip(min_shape, max_shape):
        dim_size = draw(integers(min_value=dim_min, max_value=dim_max))
        shape.append(dim_size)
    return draw(arrays(np.float32, shape, elements=dict(min_value=-1e3, max_value=1e3)))
