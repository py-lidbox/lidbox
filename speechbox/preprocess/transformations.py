"""
Transformations on datasets.
"""
import collections

import numpy as np
# import sklearn.model_selection

from speechbox import system
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

def partition_features_into_sequences(features, sequence_length):
    """Partition dataset of features into sequences."""
    for sample, onehot_label in features:
        for sequence in partition_into_sequences(sample, sequence_length):
            yield sequence, onehot_label

def speech_dataset_to_utterances(labels, paths, utterance_length_ms, utterance_offset_ms, apply_vad, print_progress, slide_over_all=True):
    """
    Iterate over all paths and labels the in given dataset group yielding utterances of specified, fixed length.
    If slide_over_all is given and False, every audio file will be concatenated and utterances yielded from a window that slides over every file, regardless of utterance boundaries.
    """
    # Working memory for incomplete utterances
    label_to_wav = {label: np.zeros((0,)) for label in set(labels)}
    for i, (label, wavpath) in enumerate(zip(labels, paths), start=1):
        # Read file using native rate
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

def utterances_to_features(utterances, label_to_index, extractors):
    """
    Iterate over utterances, extracting features from each utterance with given extractors and yield the features as sequences and corresponding labels.
    """
    assert len(extractors) > 0, "No extractors defined"
    for label, utterance in utterances:
        onehot = np.zeros(len(label_to_index), dtype=np.float32)
        onehot[label_to_index[label]] = 1.0
        # Apply first extractor
        extractor = extractors[0]
        feats = features.extract_features(utterance, extractor["name"], extractor.get("kwargs"))
        # If there are more extractors, apply them sequentially and append results to features
        #TODO extractors[1:], figuring out how to merge dimensions might get tricky
        yield feats, onehot

def utterances_to_features_no_labels(utterances, extractors):
    assert len(extractors) > 0, "No extractors defined"
    for utterance in utterances:
        extractor = extractors[0]
        return features.extract_features(utterance, extractor["name"], extractor.get("kwargs"))

def files_to_features(paths, config):
    """
    Extract utterances from all audio files in the given iterator of paths, using parameters from the given experiment config.
    """
    for path in paths:
        # One file into chunks, using dummy label None
        utterance_chunks = speech_dataset_to_utterances(
            [None], [path],
            utterance_length_ms=config["utterance_length_ms"],
            utterance_offset_ms=config["utterance_offset_ms"],
            apply_vad=config.get("apply_vad", False),
            print_progress=config.get("print_progress", 0),
        )
        # Drop dummy labels
        utterance_chunks = (utterance for _, utterance in utterance_chunks)
        # Extract features from utterance chunks
        features = utterances_to_features_no_labels(
            utterance_chunks,
            extractors=config["extractors"],
        )
        sequence_length = config.get("sequence_length", 0)
        if sequence_length > 0:
            features = transformations.partition_into_sequences(features, sequence_length)
        yield np.array(list(features))

def dataset_split_samples(samples, validation_ratio=0.10, test_ratio=0.10, random_state=None, verbosity=0):
    """
    Perform random training-validation-test split for samples.
    """
    # training-test split from whole dataset
    training_samples, test_samples = sklearn.model_selection.train_test_split(
        samples,
        random_state=random_state,
        test_size=test_ratio
    )
    # training-validation split from training set
    training_samples, validation_samples = sklearn.model_selection.train_test_split(
        training_samples,
        random_state=random_state,
        test_size=validation_ratio / (1.0 - test_ratio)
    )
    return {
        "training": training_samples,
        "validation": validation_samples,
        "test": test_samples,
    }

def dataset_split_samples_by_speaker(samples, parse_speaker_id, validation_ratio=0.10, test_ratio=0.10, random_state=None, verbosity=0):
    """
    Same as dataset_split_samples, but the split will be disjoint by speaker ID.
    In this case, test_ratio is the ratio of unique speakers in the test set to unique speakers in the training set (and similarily for the validation_ratio).
    The amount of samples per speaker should be approximately equal for all speakers to avoid inbalanced amount of samples in the resulting split.
    """
    speakers = list(set(parse_speaker_id(sample[0]) for sample in samples))
    training_speakers, test_speakers = sklearn.model_selection.train_test_split(
        speakers,
        random_state=random_state,
        test_size=test_ratio
    )
    training_speakers, validation_speakers = sklearn.model_selection.train_test_split(
        training_speakers,
        random_state=random_state,
        test_size=validation_ratio / (1.0 - test_ratio)
    )
    return {
        key: [sample for sample in samples if parse_speaker_id(sample[0]) in set(split_speakers)]
        for key, split_speakers in (
            ("training", training_speakers),
            ("validation", validation_speakers),
            ("test", test_speakers),
        )
    }

def dataset_split_parse_predefined(dataset_walker, verbosity=0):
    raise NotImplementedError("rewrite todo")
    assert hasattr(dataset_walker, "datagroup_patterns"), "The given dataset walker, '{}', does not seem to support parsing predefined datagroup splits".format(repr(dataset_walker))
    expected_datagroups = tuple(key for key, _ in dataset_walker.datagroup_patterns)
    split = {datagroup_key: collections.defaultdict(list) for datagroup_key in expected_datagroups}
    for label, wavpath in dataset_walker.walk(verbosity=verbosity):
        datagroup_key = dataset_walker.parse_datagroup(wavpath)
        if datagroup_key is None:
            error_msg = "dataset walker '{}' was unable to parse datagroup key for path '{}'".format(repr(dataset_walker), wavpath)
            assert False, error_msg
        split[datagroup_key]["labels"].append(label)
        split[datagroup_key]["paths"].append(wavpath)
    return split
