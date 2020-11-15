"""
Dataset metadata parsing/loading/preprocessing.
"""
from concurrent.futures import ThreadPoolExecutor
import collections
import itertools
import os

import miniaudio
import numpy as np
import pandas as pd


REQUIRED_META_COLUMNS = (
    "path",
    "label",
    "split",
)


def verify_integrity(meta, max_threads=os.cpu_count()):
    """
    Check that
    1. The metadata table contains all required columns.
    2. There are no NaN values.
    3. All audio filepaths exist on disk.
    4. All splits/buckets are disjoint by speaker id.

    This function throws an exception if verification fails, otherwise completes silently.
    """
    missing_columns = set(REQUIRED_META_COLUMNS) - set(meta.columns)
    assert missing_columns == set(), "{} missing columns in metadata: {}".format(len(missing_columns), sorted(missing_columns))

    assert not meta.isna().any(axis=None), "NaNs in metadata"

    if max_threads > 0:
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
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


def read_audio_durations(meta, max_threads=os.cpu_count()):
    def _get_duration(row):
        id, path = row
        return id, miniaudio.get_file_info(path).duration

    if max_threads > 0:
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            durations = list(pool.map(_get_duration, meta.path.items(), chunksize=1000))
    else:
        durations = [_get_duration(row) for row in meta.path.items()]

    #TODO this should be a test case
    assert len(durations) == len(meta.index) and all(id1 == id2 for (id1, _), id2 in zip(durations, meta.index)), "incorrect order of rows after computing audio durations"

    return np.array([d for _, d in durations], np.float32)


#TODO check if this could be replaced with some adapter to a library designed for
# imbalanced datasets that would support custom imbalance metrics/weights like durations in seconds
# in our case.
def random_oversampling(meta, copy_flag="is_copy", random_state=None):
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
    # Add flag column to distinguish copies and original rows
    if copy_flag not in meta.columns:
        meta = meta.assign({copy_flag: False})

    durations_by_label = meta[["label", "duration"]].groupby("label")

    total_dur = durations_by_label.sum()
    target_label = total_dur.idxmax()[0]
    total_dur_delta = total_dur.loc[target_label] - total_dur
    median_dur = durations_by_label.median()
    sample_sizes = (total_dur_delta / median_dur).astype(np.int32)

    copies = []

    def mark_copy(row):
        row["id"] = "{}_copy_{}".format(row["id"], row.name)
        row[copy_flag] = True
        return row

    for label in durations_by_label.groups:
        if label != target_label:
            sample_size = sample_sizes.loc[label][0]
            copy = (meta[meta["label"]==label]
                      .sample(n=sample_size, replace=True, random_state=random_state)
                      .reset_index()
                      .transform(mark_copy, axis=1))
            copies.append(copy)

    copied_meta = pd.concat(copies).set_index("id", drop=True)
    return pd.concat([copied_meta, meta], verify_integrity=True).sort_index()


def random_oversampling_on_split(meta, split):
    meta = meta.assign(is_copy=False)

    sampled = meta[meta["split"]==split]
    rest = meta[meta["split"]!=split]

    return pd.concat([random_oversampling(sampled), rest], verify_integrity=True).sort_index()


def generate_label2target(meta):
    """
    Generate a unique label-to-target mapping,
    where integer targets are the enumeration of labels in lexicographic order.
    """
    label2target = collections.OrderedDict(
            (l, t) for t, l in enumerate(sorted(meta.label.unique())))
    meta["target"] = np.array([label2target[l] for l in meta.label], np.int32)
    return meta, label2target
