"""
Dataset metadata parsing/loading/preprocessing.
"""
from concurrent.futures import ThreadPoolExecutor
import itertools
import os

import miniaudio
import numpy as np
import pandas as pd


def verify_integrity(meta):
    """
    Check that
    1. There are no NaN values.
    2. All paths exist on disk.
    3. All splits are disjoint by speaker id.
    """
    assert not meta.isna().any(axis=None), "NaNs in metadata"

    def assert_path_exists(path):
        assert os.path.exists(path), "path '{}' does not exist".format(path)

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
