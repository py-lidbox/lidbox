import collections

import librosa
import numpy as np
import webrtcvad


def remove_silence(wav):
    """
    Perform voice activity detection with webrtcvad.
    """
    frame_length_ms = 10
    expected_sample_rates = (8000, 16000, 32000, 48000)
    data, fs = wav
    assert fs in expected_sample_rates, "sample rate was {}, but webrtcvad supports only following samples rates: {}".format(fs, expected_sample_rates)
    frame_width = int(fs * frame_length_ms * 1e-3)
    # Do voice activity detection for each frame, creating an index filter containing True if frame is speech and False otherwise
    vad = webrtcvad.Vad()
    speech_indexes = []
    for frame_start in range(0, data.size - (data.size % frame_width), frame_width):
        frame_bytes = bytes(data[frame_start:(frame_start + frame_width)])
        speech_indexes.extend(frame_width*[vad.is_speech(frame_bytes, fs)])
    # Always filter out the tail if it does not fit inside the frame
    speech_indexes.extend((data.size % frame_width) * [False])
    return data[speech_indexes], fs

def extract_features(utterance_wav, extractor):
    assert isinstance(utterance_wav, tuple) and len(utterance_wav) == 2, "Expected utterance as a (signal, sample rate) tuple, but got type '{}'".format(type(utterance_wav))
    features_func = all_extractors[extractor]["callable"]
    num_features = all_extractors[extractor]["num_features"]
    features = features_func(utterance_wav)
    assert features.ndim == 2, "Unexpected dimension {} for features, expected 2".format(features.ndim)
    assert features.shape[1] == num_features, "Unexpected number of features {}, expected {}".format(features.shape[1], num_features)
    return features

def mfcc_deltas_012(utterance):
    """
    MFCCs, normalize each coef by L2 norm, compute 1st and 2nd order deltas on MFCCs and append them to the "0th order delta".
    Return an array of (0th, 1st, 2nd) deltas for each sample in utterance.
    """
    signal, rate = utterance
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13) #FIXME n_mfcc
    # Normalize each coefficient such that the L2 norm for each coefficient over all frames is equal to 1
    mfccs = librosa.util.normalize(mfccs, norm=2.0, axis=0)
    # Compute deltas and delta-deltas and interleave them for every frame
    # i.e. for frames f_i, the features are:
    # f_0, d(f_0), d(d(f_0)), f_1, d(f_1), d(d(f_1)), f_2, d(f_2), ...
    features = np.empty((3 * mfccs.shape[0], mfccs.shape[1]), dtype=mfccs.dtype)
    features[0::3] = mfccs
    features[1::3] = librosa.feature.delta(mfccs, order=1, width=3)
    features[2::3] = librosa.feature.delta(mfccs, order=2, width=5)
    return features.T


all_extractors = collections.OrderedDict({
    "mfcc-deltas-012": {
        "callable": mfcc_deltas_012,
        "num_features": 3 * 13,
    },
})
