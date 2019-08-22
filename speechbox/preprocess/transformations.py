"""
Transformations on datasets.
"""
import collections

import numpy as np
import sklearn.model_selection

import speechbox.system as system
import speechbox.preprocess.features as features


def partition_into_sequences(data, sequence_length):
    """
    Partition all rows of data into sequences.
    If the data is not divisible by the sequence length, pad the last partition with zeros.
    >>> import numpy as np
    >>> a = np.random.normal(0, 1, (11, 3))
    >>> p = partition_into_sequences(a, 2)
    >>> assert np.linalg.norm(a) == np.linalg.norm(p), "Invalid sequence partition"
    >>> assert a.shape[1] == p.shape[2], "Invalid sequence partition"
    >>> assert a.shape[0] <= p.shape[0]*p.shape[1], "Invalid sequence partition"
    """
    assert data.ndim == 2, "Unexpected dimensions for data to partition: {}, expected 2".format(data.ndim)
    num_sequences = (data.shape[0] + sequence_length - 1) // sequence_length
    # Explicit copy required since we do not own the reference to the underlying data and data.resize modifies it in-place
    resized = data.copy()
    resized.resize((num_sequences, sequence_length, data.shape[1]))
    return resized

def speech_dataset_to_utterances(labels, paths, utterance_length_ms, utterance_offset_ms, apply_vad, print_progress, slide_over_all=True):
    """
    Iterate over all paths and labels the in given dataset group yielding utterances of specified, fixed length.
    If slide_over_all is given and False, every audio file will be concatenated and utterances yielded from a window that slides over every file, regardless of utterance boundaries.
    """
    # Working memory for incomplete utterances
    label_to_wav = {label: np.zeros((0,)) for label in set(labels)}
    for i, (label, wavpath) in enumerate(zip(labels, paths), start=1):
        wav, rate = system.read_wavfile(wavpath)
        if apply_vad:
            wav, _ = system.remove_silence((wav, rate))
        # If we are merging, prepend partial utterance from end of previous file
        if slide_over_all:
            wav = np.concatenate((label_to_wav[label], wav))
        utterance_boundary = int(rate * utterance_length_ms * 1e-3)
        utterance_offset = int(rate * utterance_offset_ms * 1e-3)
        assert utterance_boundary > 0, "Invalid boundary, {}, for utterance from file '{}'".format(utterance_boundary, wavpath)
        assert utterance_offset > 0, "Invalid offset, {}, for utterance from file '{}'".format(utterance_offset, wavpath)
        # Split wav starting from index 0 into utterances of specified length and yield each utterance
        while wav.size > utterance_boundary:
            # Speech signal is long enough to produce an utterance, yield it
            yield label, (wav[:utterance_boundary], rate)
            # Move utterance window forward
            wav = wav[utterance_offset:]
        if slide_over_all:
            # Put rest back to wait for the next signal chunk
            label_to_wav[label] = wav
        else:
            # Drop tail and start from scratch for next file
            label_to_wav[label] = np.zeros((0,))
        if print_progress and i % print_progress == 0:
            print(i, "done")

def utterances_to_features(utterances, label_to_index, extractors, sequence_length):
    """
    Iterate over utterances, extracting features from each utterance with given extractors and yield the features as sequences and corresponding labels.
    """
    assert len(extractors) > 0, "No extractors defined"
    for label, utterance in utterances:
        onehot = np.zeros(len(label_to_index), dtype=np.float32)
        onehot[label_to_index[label]] = 1.0
        # Apply first extractor
        extractor = extractors[0].copy()
        feats = features.extract_features(utterance, extractor["name"], extractor.get("kwargs"))
        # If there are more extractors, apply them sequentially and append results to features
        #TODO extractors[1:], figuring out how to merge dimensions might get tricky
        sequences = partition_into_sequences(feats, sequence_length)
        for sequence in sequences:
            yield sequence, onehot

def files_to_utterances(paths, config):
    """
    Extract utterances from all audio files in the given iterator of paths, using parameters from the given experiment config.
    """
    for path in paths:
        utterance_chunks = speech_dataset_to_utterances(
            [0], [path],
            utterance_length_ms=config["utterance_length_ms"],
            utterance_offset_ms=config["utterance_offset_ms"],
            apply_vad=config.get("apply_vad", False),
            print_progress=config.get("print_progress", 0)
        )
        features = utterances_to_features(
            utterance_chunks,
            label_to_index=[0],
            extractors=config["extractors"],
            sequence_length=config["sequence_length"]
        )
        # Evaluate features generator while dropping dummy labels
        features = [feat for feat, _ in features]
        features = np.array(features) if features else None
        yield path, features

def dataset_split_samples(dataset_walker, validation_ratio=0.10, test_ratio=0.10, random_state=None, verbosity=0):
    """
    Collect all wavpaths with the given dataset_walker and perform a random training-validation-test split.
    Returns a 3-tuple of (paths, labels, checksums) pairs for each split.
    """
    all_labels, all_paths, all_checksums = tuple(zip(*dataset_walker.walk(verbosity=verbosity)))
    # training-test split from whole dataset
    training_paths, test_paths, training_labels, test_labels, training_checksums, test_checksums = sklearn.model_selection.train_test_split(
        all_paths,
        all_labels,
        all_checksums,
        random_state=random_state,
        test_size=test_ratio
    )
    # training-validation split from training set
    split = sklearn.model_selection.train_test_split(
        training_paths,
        training_labels,
        training_checksums,
        random_state=random_state,
        test_size=validation_ratio / (1.0 - test_ratio)
    )
    return {
        "training": {
            "paths": split[0],
            "labels": split[2],
            "checksums": split[4],
        },
        "validation": {
            "paths": split[1],
            "labels": split[3],
            "checksums": split[5],
        },
        "test": {
            "paths": test_paths,
            "labels": test_labels,
            "checksums": test_checksums,
        }
    }

def dataset_split_samples_by_speaker(dataset_walker, validation_ratio=0.10, test_ratio=0.10, random_state=None, verbosity=0):
    """
    Same as dataset_split_samples, but the training-set split will be disjoint by speaker ID.
    In this case, test_ratio is the ratio of unique speakers in the test set to unique speakers in the training set (and similarily for the validation_ratio).
    The amount of samples per speaker should be approximately equal for all speakers to avoid inbalanced amount of samples in the resulting split.
    """
    training_speakers = {}
    test_speakers = {}
    for label, speaker_ids in dataset_walker.speaker_ids_by_label().items():
        train_split, test_split = sklearn.model_selection.train_test_split(
            speaker_ids,
            random_state=random_state,
            test_size=test_ratio
        )
        training_speakers[label] = train_split
        test_speakers[label] = test_split
    # Set dataset_walker to return only files by training-set speaker IDs
    dataset_walker.set_speaker_filter(training_speakers)
    training_labels, training_paths, training_checksums = tuple(zip(*dataset_walker.walk(verbosity=verbosity)))
    # Perform training-validation split by sample, i.e. one speaker may or may not have samples in both sets
    split = sklearn.model_selection.train_test_split(
        training_paths,
        training_labels,
        training_checksums,
        random_state=random_state,
        test_size=validation_ratio / (1.0 - test_ratio)
    )
    # Set dataset_walker to return only files by test-set speaker IDs
    dataset_walker.set_speaker_filter(test_speakers)
    test_labels, test_paths, test_checksums = tuple(zip(*dataset_walker.walk(verbosity=verbosity)))
    return {
        "training": {
            "paths": split[0],
            "labels": split[2],
            "checksums": split[4],
        },
        "validation": {
            "paths": split[1],
            "labels": split[3],
            "checksums": split[5],
        },
        "test": {
            "paths": test_paths,
            "labels": test_labels,
            "checksums": test_checksums,
        }
    }

def dataset_split_parse_predefined(dataset_walker, verbosity=0):
    assert hasattr(dataset_walker, "datagroup_patterns"), "The given dataset walker, '{}', does not seem to support parsing predefined datagroup splits".format(repr(dataset_walker))
    expected_datagroups = tuple(key for key, _ in dataset_walker.datagroup_patterns)
    split = {datagroup_key: collections.defaultdict(list) for datagroup_key in expected_datagroups}
    for label, wavpath, md5sum in dataset_walker.walk(verbosity=verbosity):
        datagroup_key = dataset_walker.parse_datagroup(wavpath)
        if datagroup_key is None:
            error_msg = "dataset walker '{}' was unable to parse datagroup key for path '{}'".format(repr(dataset_walker), wavpath)
            assert False, error_msg
        split[datagroup_key]["labels"].append(label)
        split[datagroup_key]["paths"].append(wavpath)
        split[datagroup_key]["checksums"].append(md5sum)
    return split
