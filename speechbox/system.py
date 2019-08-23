"""File IO."""
import gzip
import hashlib
import itertools
import json
import subprocess

from audioread.exceptions import NoBackendError
import librosa
import sox
import webrtcvad
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
            stderr=subprocess.PIPE
        )
        yield process.stdout.decode("utf-8").strip()

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

def load_features_as_dataset(tfrecord_paths, training_config=None):
    import tensorflow as tf
    if training_config is None:
        training_config = {}
    # All labels should have features of same dimensions
    features_meta = load_features_meta(tfrecord_paths[0])
    assert all(features_meta == load_features_meta(record_path) for record_path in tfrecord_paths), "All labels should have features with equal dimensions"
    num_labels = features_meta["num_labels"]
    num_features =  features_meta["num_features"]
    def parse_sequence_example(seq_example_string):
        return sequence_example_to_model_input(seq_example_string, num_labels, num_features)
    def parse_compressed_tfrecords(paths):
        d = tf.data.TFRecordDataset(paths, compression_type=TFRECORD_COMPRESSION)
        if "parallel_parse" in training_config:
            d = d.map(parse_sequence_example, num_parallel_calls=training_config["parallel_parse"])
        else:
            d = d.map(parse_sequence_example)
        return d
    class_weights = training_config.get("class_weight")
    if class_weights:
        assert len(class_weights) == len(tfrecord_paths), "Amount of label draw probabilities should match amount of tfrecord files"
        # Assign a higher probability for drawing a more rare sample by inverting ratios of label to total num labels
        draw_prob = {label: 1.0/class_weight for label, class_weight in class_weights.items()}
        # Normalize so it is a prob distribution
        tot = sum(draw_prob.values())
        draw_prob = {label: inv_ratio/tot for label, inv_ratio in draw_prob.items()}
        # Assume .tfrecord files have been named by label
        weights = [
            draw_prob[os.path.basename(path).split(".tfrecord")[0]]
            for path in tfrecord_paths
        ]
    else:
        # All samples equally probable
        weights = [1.0 for path in tfrecord_paths]
    # Assume each tfrecord file contains features only for a single label
    label_datasets = [parse_compressed_tfrecords([path]) for path in tfrecord_paths]
    if "repeat" in training_config:
        label_datasets = [d.repeat(count=training_config["repeat"]) for d in label_datasets]
    dataset = tf.data.experimental.sample_from_datasets(label_datasets, weights=weights)
    if "dataset_shuffle_size" in training_config:
        dataset = dataset.shuffle(training_config["dataset_shuffle_size"])
    if "batch_size" in training_config:
        dataset = dataset.batch(training_config["batch_size"])
        if "prefetch" in training_config:
            dataset = dataset.prefetch(training_config["steps_per_epoch"])
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
    for config in transform_steps:
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

def get_total_duration(paths):
    secs = get_total_duration_sec(paths)
    mins, secs = secs // 60, secs % 60
    hours, mins = mins // 60, mins % 60
    return hours, mins, secs

def format_duration(duration):
    return "{:02d}h {:02d}min {:02d}sec".format(*duration)
