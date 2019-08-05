"""File IO."""
import hashlib
import itertools
import json

import librosa
import tensorflow as tf
import sox
import yaml


def read_wavfile(path, **librosa_kwargs):
    if "sr" not in librosa_kwargs:
        # Detect sampling rate if not specified
        librosa_kwargs["sr"] = None
    try:
        return librosa.core.load(path, **librosa_kwargs)
    except EOFError:
        return None, 0

def write_wav(wav, path):
    signal, rate = wav
    librosa.output.write_wav(path, signal, rate)

def get_samplerate(path, **librosa_kwargs):
    return librosa.core.get_samplerate(path, **librosa_kwargs)

def get_audio_type(path):
    return sox.file_info.file_type(path)

def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def sequence_to_example(sequence, onehot_label_vec):
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
        "target": tf.io.FixedLenFeature(shape=[num_labels], dtype=tf.float32),
    }
    sequence_definition = {
        "inputs": tf.io.FixedLenSequenceFeature(shape=[num_features], dtype=tf.float32)
    }
    context, sequence = tf.io.parse_single_sequence_example(
        seq_example_string,
        context_features=context_definition,
        sequence_features=sequence_definition
    )
    return sequence["inputs"], context["target"]

def write_features(sequence_features, target_path):
    target_path += ".tfrecord"
    # Peek the dimensions from the first sample
    sequence, onehot_label = next(sequence_features)
    features_meta = {
        "sequence_length": sequence.shape[0],
        "num_features": sequence.shape[1],
        "num_labels": len(onehot_label)
    }
    with open(target_path + ".meta.json", 'w') as meta_file:
        json.dump(features_meta, meta_file)
    # Put back the first sample
    sequence_features = itertools.chain([(sequence, onehot_label)], sequence_features)
    # Write all samples
    with tf.io.TFRecordWriter(target_path, options="GZIP") as record_writer:
        for sequence, onehot_label in sequence_features:
            sequence_example = sequence_to_example(sequence, onehot_label)
            record_writer.write(sequence_example.SerializeToString())
    return target_path

def load_features_as_dataset(tfrecord_paths, model_config):
    with open(tfrecord_paths[0] + ".meta.json") as f:
        features_meta = json.load(f)
    num_labels, num_features = features_meta["num_labels"], features_meta["num_features"]
    dataset = tf.data.TFRecordDataset(tfrecord_paths, compression_type="GZIP")
    dataset = dataset.map(lambda se: sequence_example_to_model_input(se, num_labels, num_features))
    if model_config.get("dataset_shuffle_size", 0):
        dataset = dataset.shuffle(model_config["dataset_shuffle_size"])
    dataset = dataset.repeat()
    dataset = dataset.batch(model_config["batch_size"])
    return dataset, features_meta

def write_utterance(utterance, basedir):
    label, (wav, rate) = utterance
    filename = hashlib.md5(bytes(wav)).hexdigest() + '.npy'
    with open(os.path.join(basedir, filename), "wb") as out_file:
        np.save(out_file, (label, (wav, rate)), allow_pickle=True, fix_imports=False)

def load_utterance(path):
    with open(path, "rb") as np_file:
        data = np.load(np_file, allow_pickle=True, fix_imports=False)
        return data[0], (data[1][0], data[1][1])

def load_utterances(basedir):
    for path in os.listdir(basedir):
        yield load_utterance(os.path.join(basedir, path))

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def count_dataset(tfrecord_paths):
    """
    Count the amount of entries in a TFRecord file by iterating over it once.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_paths, compression_type="GZIP")
    next_element = tf.data.make_one_shot_iterator(dataset).get_next()
    num_elements = 0
    with tf.Session() as session:
        try:
            while True:
                session.run(next_element)
                num_elements += 1
        except tf.errors.OutOfRangeError:
            # Iterator exhausted
            pass
    return num_elements
