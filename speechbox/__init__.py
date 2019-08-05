"""
Toolbox containing various speech data analysis tools.
"""
import os

def _get_unittest_data_dir():
    from speechbox import __path__
    speechbox_root = os.path.dirname(__path__[0])
    return os.path.join(speechbox_root, "test", "data_common_voice")

def _get_random_wav():
    from random import choice
    from speechbox.system import read_wavfile, get_audio_type
    data_root = _get_unittest_data_dir()
    label_dirs = [d for d in os.scandir(data_root) if d.is_dir()]
    label_dir = choice(label_dirs)
    wavpaths = [f.path for f in os.scandir(label_dir.path) if f.is_file() and get_audio_type(f.path) == "wav"]
    wavpath = choice(wavpaths)
    return label_dir.name, wavpath, read_wavfile(wavpath)

def _get_random_wav_with_mfcc():
    from speechbox.preprocess.features import extract_features
    label, wavpath, wav = _get_random_wav()
    return label, wavpath, wav, extract_features(wav, 'mfcc-deltas-012', {"n_mfcc": 13})
