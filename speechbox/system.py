"""File IO."""
import hashlib

import librosa


def read_wavfile(path, **librosa_kwargs):
    try:
        return librosa.core.load(path, **librosa_kwargs)
    except EOFError:
        return None, 0

def get_samplerate(path, **librosa_kwargs):
    return librosa.core.get_samplerate(path, **librosa_kwargs)

def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
