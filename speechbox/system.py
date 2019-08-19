"""File IO."""
import functools
import gzip
import hashlib
import itertools
import json
import subprocess

from audioread.exceptions import NoBackendError
import librosa
import sox
import yaml


TFRECORD_COMPRESSION = "GZIP"
SUBPROCESS_BATCH_SIZE = 5000

def run_for_files(cmd, filepaths):
    # Run in batches
    for begin in range(0, len(filepaths), SUBPROCESS_BATCH_SIZE):
        batch = ' '.join(filepaths[begin:begin+SUBPROCESS_BATCH_SIZE])
        process = subprocess.run(
            (cmd + ' ' + batch).split(' '),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        yield process.stdout.strip()

def read_wavfile(path, **librosa_kwargs):
    if "sr" not in librosa_kwargs:
        # Detect sampling rate if not specified
        librosa_kwargs["sr"] = None
    try:
        return librosa.core.load(path, **librosa_kwargs)
    except (EOFError, NoBackendError):
        return None, 0

def write_wav(wav, path):
    signal, rate = wav
    librosa.output.write_wav(path, signal, rate)

def get_samplerate(path, **librosa_kwargs):
    return librosa.core.get_samplerate(path, **librosa_kwargs)

def get_audio_type(path):
    try:
        return sox.file_info.file_type(path)
    except sox.core.SoxiError:
        return None

@functools.lru_cache(maxsize=2**16)
def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_gzip_json(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        return json.load(f)

def dump_gzip_json(data, path):
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(data, sort_keys=True, indent=2).encode('utf-8'))

def load_audiofile_paths(pathlist_file):
    with open(pathlist_file) as f:
        for line in f:
            line = line.strip()
            wav, _ = read_wavfile(line)
            if wav is not None:
                yield line

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

def write_features(sequence_features, target_path):
    import tensorflow as tf
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
    with tf.io.TFRecordWriter(target_path, options=TFRECORD_COMPRESSION) as record_writer:
        for sequence, onehot_label in sequence_features:
            sequence_example = sequence_to_example(sequence, onehot_label)
            record_writer.write(sequence_example.SerializeToString())
    return target_path

def load_features_meta(tfrecord_path):
    with open(tfrecord_path + ".meta.json") as f:
        return json.load(f)

def load_features_as_dataset(tfrecord_paths, model_config=None):
    import tensorflow as tf
    if model_config is None:
        model_config = {}
    # All labels should have features of same dimensions
    features_meta = load_features_meta(tfrecord_paths[0])
    assert all(features_meta == load_features_meta(record_path) for record_path in tfrecord_paths), "All labels should have features with equal dimensions"
    num_labels = features_meta["num_labels"]
    num_features =  features_meta["num_features"]
    def parse_sequence_example(seq_example_string):
        return sequence_example_to_model_input(seq_example_string, num_labels, num_features)
    def parse_compressed_tfrecords(paths):
        d = tf.data.TFRecordDataset(paths, compression_type=TFRECORD_COMPRESSION)
        if "parallel_parse" in model_config:
            d = d.map(parse_sequence_example, num_parallel_calls=model_config["parallel_parse"])
        else:
            d = d.map(parse_sequence_example)
        return d
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)
    # Consume TFRecords in cycles over all labels, each time taking a single sample with a different label
    dataset = dataset.interleave(parse_compressed_tfrecords, cycle_length=num_labels, block_length=1)
    if "dataset_shuffle_size" in model_config:
        dataset = dataset.shuffle(model_config["dataset_shuffle_size"])
    if "repeat" in model_config:
        dataset = dataset.repeat(count=model_config["repeat"])
    if "batch_size" in model_config:
        dataset = dataset.batch(model_config["batch_size"])
        if "prefetch" in model_config:
            dataset = dataset.prefetch(model_config["steps_per_epoch"])
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

def apply_sox_transformer(src_paths, dst_paths, **config):
    t = sox.Transformer()
    if "normalize" in config:
        db = float(config["normalize"])
        t = t.norm(db)
    if "volume" in config:
        amplitude = float(config["volume"])
        t = t.vol(amplitude, gain_type="amplitude")
    if "speed" in config:
        factor = float(config["speed"])
        t = t.speed(factor)
    if "reverse" in config and config["reverse"]:
        t = t.reverse()
    # Try to apply transformation on every src_path, building output files into every dst_path
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
