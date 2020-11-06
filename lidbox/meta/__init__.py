"""
Dataset metadata parsing/loading/preprocessing.
"""
from concurrent.futures import ThreadPoolExecutor
import itertools
import os

import miniaudio
import numpy as np
import pandas as pd


def verify_integrity(meta, use_threads=True):
    """
    Check that
    1. There are no NaN values.
    2. All paths exist on disk.
    3. All splits are disjoint by speaker id.
    """
    assert not meta.isna().any(axis=None), "NaNs in metadata"

    if use_threads:
        with ThreadPoolExecutor(max_workers=None) as pool:
            num_invalid = sum(int(not ok) for ok in
                              pool.map(os.path.exists, meta.path, chunksize=100))
    else:
        num_invalid = sum(int(not os.path.exists(path)) for path in meta.path)
    assert num_invalid == 0, "{} paths did not exist".format(num_invalid)

    split_names = meta.split.unique()
    split2spk = {split: set(meta[meta["split"]==split].client_id.unique())
                 for split in split_names}

    for a, b in itertools.combinations(split_names, 2):
        intersection = split2spk[a] & split2spk[b]
        assert intersection == set(), "{} and {} have {} speakers in common".format(a, b, len(intersection))


def get_mp3_duration(row):
    id, path = row
    return id, miniaudio.mp3_get_file_info(path).duration

def get_wav_duration(row):
    id, path = row
    return id, miniaudio.get_file_info(path).duration

def read_audio_durations(meta, filetype=None, use_threads=True):
    get_duration_fn = get_mp3_duration if filetype == "mp3" else get_wav_duration

    if use_threads:
        with ThreadPoolExecutor(max_workers=None) as pool:
            durations = list(pool.map(get_duration_fn, meta.path.items(), chunksize=100))
    else:
        durations = [get_duration_fn(row) for row in meta.path.items()]

    assert all(id1 == id2 for (id1, _), id2 in zip(durations, meta.index)), "incorrect order of rows after computing audio durations"

    meta["duration_sec"] = np.array([d for _, d in durations], np.float32)
    return meta


#TODO check if this could be replaced with some adapter to a library designed for
# imbalanced datasets that would support custom imbalance metrics/weights like durations in seconds
# in our case.
def random_oversampling(meta):
    """
    Random oversampling by duplicating metadata rows.

    Procedure:
    1. Select target label according to maximum total amount of speech in seconds.
    2. Compute differences in total durations between the target label and the other labels.
    3. Compute median signal length by label.
    4. Compute sample sizes by dividing the duration deltas with median signal lengths, separately for each label.
    5. Draw samples with replacement from the metadata separately for each label.
    6. Merge samples with rest of the metadata and verify there are no duplicate ids.
    """
    durations_by_label = meta[["label", "duration"]].groupby("label")

    total_dur = durations_by_label.sum()
    target_label = total_dur.idxmax()[0]
    total_dur_delta = total_dur.loc[target_label] - total_dur
    median_dur = durations_by_label.median()
    sample_sizes = (total_dur_delta / median_dur).astype(np.int32)

    samples = []

    def update_sample_id(row):
        row["id"] = "{}_copy_{}".format(row["id"], row.name)
        return row

    for label in durations_by_label.groups:
        sample_size = sample_sizes.loc[label][0]
        sample = (meta[meta["label"]==label]
                  .sample(n=sample_size, replace=True)
                  .reset_index()
                  .transform(update_sample_id, axis=1))
        samples.append(sample)

    return pd.concat(samples).set_index("id", drop=True, verify_integrity=True)
