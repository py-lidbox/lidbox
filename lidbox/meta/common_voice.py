"""
Mozilla Common Voice https://voice.mozilla.org/en/datasets
"""
import multiprocessing
import os

import pandas as pd


SPLIT_NAMES = ("train", "dev", "test")


def load(corpus_dir, lang, usecols=("client_id", "path", "sentence")):
    """
    Load metadata tsv-files of a Common Voice dataset from disk into a single pandas.DataFrame.
    """
    split_dfs = []

    for split in SPLIT_NAMES:
        df = pd.read_csv(os.path.join(corpus_dir, lang, split + ".tsv"), sep='\t', usecols=usecols)
        df = df.assign(label=lang, split=split, id='')
        df = df.transform(lambda row: fix_row(row, corpus_dir), axis=1)
        split_dfs.append(df)

    # Concatenate all split dataframes into a single table,
    # replace default integer indexing by utterance ids,
    # throwing an exception if there are duplicate utterance ids.
    return (pd.concat(split_dfs)
            .set_index("id", drop=True, verify_integrity=True)
            .sort_index())


def fix_row(row, corpus_dir):
    # Extract utterance id from mp3 clip name
    row.id = "{:s}".format(row.path.split(".mp3", 1)[0])
    # Expand path for mp3 clip
    row.path = os.path.join(corpus_dir, row.label, "clips", row.path)
    # Add language label prefix to client id to avoid id collisions with other datasets
    row.client_id = row.label + "_" + row.client_id
    return row


def load_all(corpus_dir, langs, num_processes=os.cpu_count()):
    """
    Load metadata from multiple datasets into a single table with unique utterance ids for every row.
    """
    if num_processes > 0:
        with multiprocessing.Pool(processes=num_processes) as pool:
            lang_dfs = pool.starmap(load, ((corpus_dir, lang) for lang in langs))
    else:
        lang_dfs = (load(corpus_dir, lang) for lang in langs)
    return (pd.concat(lang_dfs, verify_integrity=True).sort_index())
