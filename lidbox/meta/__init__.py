"""
Dataset metadata parsing/loading/preprocessing.
"""
from concurrent.futures import ThreadPoolExecutor
import itertools
import os

import miniaudio
import numpy as np
import pandas as pd


def assert_path_exists(path):
    assert os.path.exists(path), "integrity check failed, '{}' does not exist".format(path)


def verify_integrity(meta):
    """
    Check that
    1. All paths exist on disk.
    2. All splits are disjoint by speaker id.
    """
    with ThreadPoolExecutor(max_workers=None) as pool:
        for _, row in meta.iterrows():
            #TODO the output might look ugly on failure
            pool.submit(assert_path_exists, row["path"])

    split_names = meta.split.unique()
    split2spk = {split: set(meta[meta["split"]==split].client_id.unique())
                 for split in split_names}

    for a, b in itertools.combinations(split_names, 2):
        intersection = split2spk[a] & split2spk[b]
        assert intersection == set(), "{} and {} have {} speakers in common".format(a, b, len(intersection))


def add_mp3_durations(meta):
    meta["duration_sec"] = np.array(
        [miniaudio.mp3_get_file_info(path).duration for path in meta.path],
        np.float32)
    return meta

def add_durations(meta):
    meta["duration_sec"] = np.array(
        [miniaudio.get_file_info(path).duration for path in meta.path],
        np.float32)
    return meta


def add_audio_durations_in_parallel(meta, filetype=None):
    # https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    meta_row_chunks = np.array_split(meta, os.cpu_count())

    with ThreadPoolExecutor(max_workers=None) as pool:
        result_chunks = pool.map(
                add_mp3_durations if filetype == "mp3" else add_durations,
                meta_row_chunks)

    return pd.concat(result_chunks, verify_integrity=True).sort_index()
