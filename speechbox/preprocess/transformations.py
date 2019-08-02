"""
Transformations on datasets.
"""
import tensorflow as tf
import sklearn.model_selection


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

def sequence_to_example(sequence, onehot_label):
    """
    Encode a single sequence and its label as a TensorFlow SequenceExample.
    """
    def float_vec_to_float_features(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def sequence_to_floatlist_features(seq):
        float_features = (tf.train.Feature(float_list=tf.train.FloatList(value=frame)) for frame in seq)
        return tf.train.FeatureList(feature=float_features)
    # Time-independent context for time-dependent sequence
    context_definition = {
        "target": float_vec_to_float_features(onehot_label_vec),
    }
    context = tf.train.Features(feature=context_definition)
    # Sequence frames as a feature list
    feature_list_definition = {
        "inputs": sequence_to_floatlist_features(sequence),
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list_definition)
    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)

def sequence_example_to_model_input(seq_example_string, num_labels, num_features):
    """
    Decode a single sequence example string as an (input, target) pair to be fed into a model being trained.
    """
    context_definition = {
        "target": tf.FixedLenFeature(shape=[num_labels], dtype=tf.float32),
    }
    sequence_definition = {
        "inputs": tf.FixedLenSequenceFeature(shape=[num_features], dtype=tf.float32)
    }
    context, sequence = tf.io.parse_single_sequence_example(
        seq_example_string,
        context_features=context_definition,
        sequence_features=sequence_definition
    )
    return sequence["inputs"], context["target"]

def speech_dataset_to_utterances(dataset_walker, utterance_length_ms, utterance_offset_ms):
    """
    Iterate over dataset_walker yielding utterances of specified, fixed length.
    This can be used to transform a dataset containing audio files of arbitrary length into fixed length chunks.
    """
    label_to_wav = {label: np.empty((0,)) for label in dataset_walker.label_definitions}
    for label, wavpath in dataset_walker:
        wav, rate = dataset_walker.load(wavpath)
        wav, _ = remove_silence((wav, rate))
        wav = np.concatenate((label_to_wav[label], wav))
        utterance_boundary = int(utterance_length_ms / 1000 * rate)
        utterance_offset = int(utterance_offset_ms / 1000 * rate)
        assert utterance_boundary > 0, "Invalid boundary, {}, for utterance".format(utterance_boundary)
        assert utterance_offset > 0, "Invalid offset, {}, for utterance".format(utterance_offset)
        # Split wav starting from index 0 into utterances of specified length and yield each utterance
        while wav.size > utterance_boundary:
            # Speech signal is long enough to produce an utterance, yield it
            yield label, (wav[:utterance_boundary], rate)
            # Move utterance window forward
            wav = wav[utterance_offset:]
        # Put rest back to wait for the next signal chunk
        label_to_wav[label] = wav

def utterances_to_features(utterances, extractor, sequence_length, label_to_index):
    """
    Iterate over utterances, extracting features from each utterance with a given extractor and yield the features as sequences and corresponding labels.
    """
    for label, utterance in utterances:
        feats = features.extract_features(utterance, extractor)
        sequences = partition_into_sequences(feats, sequence_length)
        for sequence in sequences:
            onehot = np.zeros(len(label_to_index), dtype=np.float32)
            onehot[label_to_index[label]] = 1.0
            yield sequence, label, onehot

def dataset_split_samples(dataset_walker, validation_ratio=0.05, test_ratio=0.05, random_state=None):
    """
    Collect all wavpaths with the given dataset_walker and perform a random training-validation-test split.
    Returns a 3-tuple of (paths, labels) pairs:
      (
        (training_paths, training_labels),
        (validation_paths, validation_labels),
        (test_paths, test_labels)
     )
    """
    all_labels, all_paths = tuple(zip(*iter(dataset_walker)))
    # training-test split from whole dataset
    training_paths, test_paths, training_labels, test_labels = sklearn.model_selection.train_test_split(
        all_paths,
        all_labels,
        random_state=random_state,
        test_size=test_ratio
    )
    # training-validation split from training set
    training_paths, validation_paths, training_labels, validation_labels = sklearn.model_selection.train_test_split(
        training_paths,
        training_labels,
        random_state=random_state,
        test_size=validation_ratio / (1.0 - test_ratio)
    )
    return (
        (training_paths, training_labels),
        (validation_paths, validation_labels),
        (test_paths, test_labels)
    )

def dataset_split_samples_by_speaker(dataset_walker, validation_ratio=0.05, test_ratio=0.05, random_state=None):
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
    training_labels, training_paths = tuple(zip(*iter(dataset_walker)))
    # Perform training-validation split by sample, i.e. one speaker may or may not have samples in both sets
    training_paths, validation_paths, training_labels, validation_labels = sklearn.model_selection.train_test_split(
        training_paths,
        training_labels,
        random_state=random_state,
        test_size=validation_ratio / (1.0 - test_ratio)
    )
    # Set dataset_walker to return only files by test-set speaker IDs
    dataset_walker.set_speaker_filter(test_speakers)
    test_labels, test_paths = tuple(zip(*iter(dataset_walker)))
    return (
        (training_paths, training_labels),
        (validation_paths, validation_labels),
        (test_paths, test_labels)
    )
