"""File IO."""
import gzip
import hashlib
import itertools
import json
import os
import subprocess

from audioread.exceptions import NoBackendError
from scipy.io import arff
import librosa
import numpy as np
import sox
import webrtcvad
import yaml

import speechbox.preprocess.transformations as transformations


TFRECORD_COMPRESSION = "GZIP"
SUBPROCESS_BATCH_SIZE = 5000

def run_command(cmd):
    process = subprocess.run(
        cmd.split(" "),
        check=True,
        stdout=subprocess.PIPE
    )
    return process.stdout.decode("utf-8").rstrip()

def run_for_files(cmd, filepaths, batch_size=SUBPROCESS_BATCH_SIZE):
    # Run in batches
    for begin in range(0, len(filepaths), batch_size):
        batch = ' '.join(filepaths[begin:begin+batch_size])
        yield run_command(cmd + ' ' + batch)

def read_wavfile(path, **librosa_kwargs):
    if "sr" not in librosa_kwargs:
        # Detect sampling rate if not specified
        librosa_kwargs["sr"] = None
    try:
        return librosa.core.load(path, **librosa_kwargs)
    except (EOFError, NoBackendError):
        return None, 0

def read_arff_features(path, include_keys=None, exclude_keys=None, types=None):
    if types is None:
        types = {"numeric"}
    if exclude_keys is None:
        exclude_keys = {"frameTime"}
    data, meta = arff.loadarff(path)
    keys = [
        key for key, type in zip(meta.names(), meta.types())
        if (include_keys is None or key in include_keys) and key not in exclude_keys and type in types
    ]
    assert all(data[key].shape == data[keys[0]].shape for key in keys), "inconsistent dimensions in arff file, expected all to have shape {}".format(data[keys[0]].shape)
    feats = np.vstack([data[key] for key in keys if not np.any(np.isnan(data[key]))])
    return feats.T, keys

def write_wav(path, wav):
    signal, rate = wav
    librosa.output.write_wav(path, signal, rate)

def get_samplerate(path, **librosa_kwargs):
    return librosa.core.get_samplerate(path, **librosa_kwargs)

def get_audio_type(path):
    try:
        return sox.file_info.file_type(path)
    except sox.core.SoxiError:
        return None

def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def all_md5sums(paths, num_workers=32):
    from multiprocessing import Pool
    with Pool(num_workers) as pool:
        return pool.map(md5sum, paths)

def load_gzip_json(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        return json.load(f)

def dump_gzip_json(data, path):
    with gzip.open(path, "wb") as f:
        json_str = json.dumps(data, sort_keys=True, indent=2)
        f.write(json_str.encode("utf-8"))

def append_json(data, path):
    if os.path.exists(path):
        with open(path) as f:
            data_list = json.load(f)
    else:
        data_list = []
    data_list.append(data)
    with open(path, "w") as f:
        json.dump(data_list, f)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

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

def load_audiofile_paths(pathlist_file):
    with open(pathlist_file) as f:
        for line in f:
            split = line.split()
            wavpath, rest = split[0].strip(), split[1:]
            wav, _ = read_wavfile(wavpath)
            if wav is not None:
                yield wavpath, rest

def concatenate_wavs(wavs):
    assert len(wavs) > 0, "Nothing to concatenate"
    assert all(rate == wavs[0][1] for _, rate in wavs), "Cannot concatenate wavfiles with different sampling rates"
    rate = wavs[0][1]
    return np.concatenate([wav for wav, _ in wavs]), rate

def get_most_recent_file(directory):
    # Get path object with greatest unix timestamp
    files = (f for f in os.scandir(directory) if f.is_file())
    return max(files, key=lambda d: d.stat().st_mtime).name

def sequence_to_example(sequence, onehot_label_vec):
    """
    Encode a single sequence and its label as a TensorFlow SequenceExample.
    """
    import tensorflow as tf
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
    import tensorflow as tf
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

def write_sequence_features(features, target_path, sequence_length):
    import tensorflow as tf
    sequence_features = transformations.partition_features_into_sequences(features, sequence_length)
    # Peek the dimensions from the first sample
    sequence, onehot_label = next(sequence_features)
    features_meta = {
        "sequence_length": sequence.shape[0],
        "num_features": sequence.shape[1],
        "num_labels": len(onehot_label)
    }
    target_path += ".tfrecord"
    with open(target_path + ".meta.json", 'w') as meta_file:
        json.dump(features_meta, meta_file)
        meta_file.write("\n")
    # Put back the first sample
    sequence_features = itertools.chain([(sequence, onehot_label)], sequence_features)
    # Write all samples
    with tf.io.TFRecordWriter(target_path, options=TFRECORD_COMPRESSION) as record_writer:
        for sequence, onehot_label in sequence_features:
            sequence_example = sequence_to_example(sequence, onehot_label)
            record_writer.write(sequence_example.SerializeToString())
    return target_path

def features_to_example(features, onehot_label_vec):
    import tensorflow as tf
    def float_vec_to_float_features(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    features_definition = {
        "input": float_vec_to_float_features(features),
        "target": float_vec_to_float_features(onehot_label_vec),
    }
    return tf.train.Example(features=tf.train.Features(feature=features_definition))

def example_to_model_input(example_string, num_labels, num_features):
    import tensorflow as tf
    features_definition = {
        "input": tf.io.FixedLenFeature(shape=[num_features], dtype=tf.float32),
        "target": tf.io.FixedLenFeature(shape=[num_labels], dtype=tf.float32),
    }
    example = tf.io.parse_single_example(example_string, features_definition)
    return example["input"], example["target"]

def write_features(features, target_path):
    import tensorflow as tf
    target_path += ".tfrecord"
    feat, onehot_label = next(features)
    assert feat.ndim == 1, "Unexpected dimensions '{}' for 1-dim vector features".format(feat.ndim)
    features_meta = {
        "num_features": feat.size,
        "num_labels": len(onehot_label)
    }
    with open(target_path + ".meta.json", 'w') as meta_file:
        json.dump(features_meta, meta_file)
        meta_file.write("\n")
    features = itertools.chain([(feat, onehot_label)], features)
    with tf.io.TFRecordWriter(target_path, options=TFRECORD_COMPRESSION) as record_writer:
        for feat, onehot_label in features:
            example = features_to_example(feat, onehot_label)
            record_writer.write(example.SerializeToString())
    return target_path

def count_all_features(features_file):
    from tensorflow import device
    with device("/CPU:0"):
        dataset, meta = load_features_as_dataset([features_file])
        return int(dataset.reduce(0, lambda count, _: count + 1)), meta

def count_all_features_parallel(labels, features_files, num_workers=None):
    from multiprocessing import Pool
    assert len(labels) == len(features_files)
    if num_workers is None:
        num_workers = len(features_files)
    with Pool(num_workers) as pool:
        return zip(labels, pool.map(count_all_features, features_files))

def load_features_meta(tfrecord_path):
    with open(tfrecord_path + ".meta.json") as f:
        return json.load(f)

def load_features_as_dataset(tfrecord_paths, training_config=None):
    import tensorflow as tf
    if training_config is None:
        training_config = {}
    # All labels should have features of same dimensions
    features_meta = load_features_meta(tfrecord_paths[0])
    assert all(features_meta == load_features_meta(record_path) for record_path in tfrecord_paths), "All labels should have features with equal dimensions"
    num_labels = features_meta["num_labels"]
    num_features =  features_meta["num_features"]
    if features_meta.get("sequence_length", 0) > 0:
        example_parser_fn = lambda example_str: sequence_example_to_model_input(example_str, num_labels, num_features)
    else:
        example_parser_fn = lambda example_str: example_to_model_input(example_str, num_labels, num_features)
    def parse_compressed_tfrecords(paths):
        d = tf.data.TFRecordDataset(paths, compression_type=TFRECORD_COMPRESSION)
        if "parallel_parse" in training_config:
            d = d.map(example_parser_fn, num_parallel_calls=training_config["parallel_parse"])
        else:
            d = d.map(example_parser_fn)
        return d
    label_weights = training_config.get("label_weights")
    if label_weights:
        assert len(label_weights) == len(tfrecord_paths), "Amount of label draw probabilities should match amount of tfrecord files"
        # Assign a higher probability for drawing a more rare sample by inverting ratios of label to total num labels
        draw_prob = {label: 1.0/w for label, w in label_weights.items()}
        # Normalize into a probability distribution
        tot = sum(draw_prob.values())
        draw_prob = {label: inv_w/tot for label, inv_w in draw_prob.items()}
        # Assume .tfrecord files have been named by label
        weights = [
            draw_prob[os.path.basename(path).split(".tfrecord")[0]]
            for path in tfrecord_paths
        ]
        # Assume each tfrecord file contains features only for a single label
        label_datasets = [parse_compressed_tfrecords([path]) for path in tfrecord_paths]
        if "repeat" in training_config:
            label_datasets = [d.repeat(count=training_config["repeat"]) for d in label_datasets]
        dataset = tf.data.experimental.sample_from_datasets(label_datasets, weights=weights)
    else:
        dataset = parse_compressed_tfrecords(tfrecord_paths)
        if "repeat" in training_config:
            dataset = dataset.repeat(count=training_config["repeat"])
    if "shuffle_buffer_size" in training_config:
        dataset = dataset.shuffle(training_config["shuffle_buffer_size"])
    if "batch_size" in training_config:
        dataset = dataset.batch(training_config["batch_size"])
    if "prefetch" in training_config:
        dataset = dataset.prefetch(training_config["prefetch"])
    return dataset, features_meta

def iter_log_events(tf_event_file):
    import tensorflow as tf
    from tensorflow.core.util.event_pb2 import Event
    for event in tf.data.TFRecordDataset([tf_event_file]):
        event = Event.FromString(event.numpy())
        if event.summary.value:
            assert len(event.summary.value) == 1, "Unexpected length for event summary"
            value = event.summary.value[0]
            yield value.tag, value.simple_value

def remove_silence(wav, aggressiveness=0):
    """
    Perform voice activity detection with webrtcvad.
    """
    frame_length_ms = 10
    expected_sample_rates = (8000, 16000, 32000, 48000)
    data, fs = wav
    assert fs in expected_sample_rates, "sample rate was {}, but webrtcvad supports only following samples rates: {}".format(fs, expected_sample_rates)
    frame_width = int(fs * frame_length_ms * 1e-3)
    # Do voice activity detection for each frame, creating an index filter containing True if frame is speech and False otherwise
    vad = webrtcvad.Vad(aggressiveness)
    speech_indexes = []
    for frame_start in range(0, data.size - (data.size % frame_width), frame_width):
        frame_bytes = bytes(data[frame_start:(frame_start + frame_width)])
        speech_indexes.extend(frame_width*[vad.is_speech(frame_bytes, fs)])
    # Always filter out the tail if it does not fit inside the frame
    speech_indexes.extend((data.size % frame_width) * [False])
    return data[speech_indexes], fs

def apply_sox_transformer(src_paths, dst_paths, transform_steps):
    t = sox.Transformer()
    for transform, value in transform_steps:
        if transform == "normalize":
            t = t.norm(float(value))
        elif transform == "volume":
            t = t.vol(float(value), gain_type="amplitude")
        elif transform == "speed":
            t = t.speed(float(value))
        elif transform == "reverse" and value:
            t = t.reverse()
    # Try to apply the transformation on every src_path, building output files into every dst_path
    for src, dst in zip(src_paths, dst_paths):
        if t.build(src, dst):
            yield src, dst
        else:
            yield src, None

def get_total_duration_sec(paths):
    # Run SoXi for all files
    soxi_cmd = "soxi -D -T"
    seconds = sum(float(output) for output in run_for_files(soxi_cmd, paths))
    return round(seconds)

def get_total_duration(paths):
    secs = get_total_duration_sec(paths)
    mins, secs = secs // 60, secs % 60
    hours, mins = mins // 60, mins % 60
    return hours, mins, secs

def format_duration(duration):
    return "{:02d}h {:02d}min {:02d}sec".format(*duration)

def parse_path_list(path):
    paths = []
    labels = []
    with open(path) as f:
        for line in f:
            path, label = line.strip().split()[:2]
            paths.append(path)
            labels.append(label)
    return paths, labels
