"""
Toolbox containing various speech data analysis tools.
"""
import os
import yaml

def get_package_root():
    from speechbox import __path__
    return os.path.abspath(os.path.dirname(__path__[0]))

def _get_unittest_data_dir():
    datadir = os.path.join(get_package_root(), "test", "acoustic_data")
    assert os.path.isdir(datadir), "Acoustic data has not yet been downloaded to '{}'".format(datadir)
    return datadir

def _get_random_wav():
    from random import choice
    from speechbox.system import read_wavfile, get_audio_type
    data_root = _get_unittest_data_dir()
    label_dirs = [d for d in os.scandir(data_root) if d.is_dir()]
    label_dir = choice(label_dirs)
    wavpaths = [f.path for f in os.scandir(os.path.join(label_dir.path, "wav")) if f.is_file() and get_audio_type(f.path) == "wav"]
    wavpath = choice(wavpaths)
    return label_dir.name, wavpath, read_wavfile(wavpath)

def yaml_pprint(d, **kwargs):
    print(yaml.dump(d, indent=4), **kwargs)
