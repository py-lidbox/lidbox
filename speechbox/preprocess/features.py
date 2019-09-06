import collections

import librosa
import numpy as np


def extract_features(utterance_wav, extractor, config):
    assert isinstance(utterance_wav, tuple) and len(utterance_wav) == 2, "Expected utterance as a (signal, sample_rate) tuple, but got type '{}'".format(type(utterance_wav))
    extractor_func = all_extractors[extractor]
    features = extractor_func(utterance_wav, **config)
    assert features.ndim == 2, "Unexpected dimension {} for features, expected 2".format(features.ndim)
    return features

def mfcc(utterance, normalize=True, **kwargs):
    """
    MFCCs and normalize each coefficient to 0 mean and 1 variance.
    """
    signal, rate = utterance
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, **kwargs)
    # Normalize each coefficient to 0 mean and 1 variance
    if normalize:
        mfccs = (mfccs - mfccs.mean(axis=0)) / mfccs.std(axis=0)
    return mfccs.T

def mfcc_deltas_012(utterance, **kwargs):
    """
    MFCCs, normalize, compute 1st and 2nd order deltas on MFCCs and append them to the "0th order delta".
    Return an array of (0th, 1st, 2nd) deltas for each sample in utterance.
    """
    mfccs = mfcc(utterance, **kwargs).T
    # Compute deltas and delta-deltas and interleave them for every frame
    # i.e. for frames f_i, the features are:
    # f_0, d(f_0), d(d(f_0)), f_1, d(f_1), d(d(f_1)), f_2, d(f_2), ...
    features = np.empty((3 * mfccs.shape[0], mfccs.shape[1]), dtype=mfccs.dtype)
    features[0::3] = mfccs
    features[1::3] = librosa.feature.delta(mfccs, order=1, width=3)
    features[2::3] = librosa.feature.delta(mfccs, order=2, width=5)
    return features.T

def sdc(frames, d, P, k, **kwargs):
    """
    Shifted-delta coefficients from
    Torres-Carrasquillo, P. A. el al. (2002).
    https://www.isca-speech.org/archive/icslp_2002/i02_0089.html

    N is assumed to be frames.shape[1].
    """
    assert d > 0 and P > 0 and k > 0, "invalid sdc parameters d = {}, P = {}, k = {}".format(d, P, k)
    assert frames.ndim > 1, "invalid dimensions for feature frames: {}".format(frames.ndim)
    num_frames = frames.shape[0]
    frames = np.pad(frames, pad_width=((0, d + k*P), (0, 0)), mode='constant', constant_values=0)
    #TODO without loop expression
    return np.concatenate([frames[i+2*d::P][:k] - frames[i::P][:k] for i in range(num_frames)])

all_extractors = collections.OrderedDict([
    ("mfcc", mfcc),
    ("mfcc-deltas-012", mfcc_deltas_012),
    ("sdc", sdc),
])
