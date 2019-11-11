import tensorflow as tf
from . import librosa_tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_COMPRESSION = "GZIP"

def floats2floatlist(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def sequence2floatlists(v_seq):
    return tf.train.FeatureList(feature=map(floats2floatlist, v_seq))

def string2byteslist(s):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.numpy()]))

def serialize_sequence_features(features, meta):
    context_definition = {
        "uuid": string2byteslist(meta[0]),
        "label": string2byteslist(meta[1]),
    }
    context = tf.train.Features(feature=context_definition)
    sequence_definition = {
        "features": sequence2floatlists(features),
    }
    feature_lists = tf.train.FeatureLists(feature_list=sequence_definition)
    seq_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return seq_example.SerializeToString()

def deserialize_sequence_features(seq_example_str, feature_dim):
    context_definition = {
        "uuid": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[1], dtype=tf.string),
    }
    sequence_definition = {
        "features": tf.io.FixedLenSequenceFeature(shape=[feature_dim], dtype=tf.float32)
    }
    context, sequence = tf.io.parse_single_sequence_example(
        seq_example_str,
        context_features=context_definition,
        sequence_features=sequence_definition
    )
    return sequence["features"], context

def write_features(extractor_dataset, target_path):
    serialize = lambda feats, meta: tf.py_function(serialize_sequence_features, (feats, meta), tf.string)
    record_writer = tf.data.experimental.TFRecordWriter(target_path, compression_type=TFRECORD_COMPRESSION)
    record_writer.write(extractor_dataset.map(serialize, num_parallel_calls=AUTOTUNE))

def load_features(tfrecord_paths, feature_dim, dataset_config):
    deserialize = lambda s: deserialize_sequence_features(s, feature_dim)
    ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type=TFRECORD_COMPRESSION)
    return ds.map(deserialize)

def prepare_dataset_for_training(ds, config, label2onehot):
    rnn_steps = config.get("rnn_steps")
    if rnn_steps:
        seq_len, seq_step = rnn_steps["frame_length"], rnn_steps["frame_step"]
        make_timesteps_frames = lambda feats, meta: tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(tf.signal.frame(feats, seq_len, seq_step, axis=0)),
            tf.data.Dataset.from_tensors(meta).repeat(-1)))
        ds = ds.flat_map(make_timesteps_frames)
    to_model_input = lambda feats, meta: (feats, label2onehot(meta[1]))
    ds = ds.map(to_model_input)
    shuffle_size = config.get("shuffle_buffer_size", 0)
    if shuffle_size:
        ds = ds.shuffle(shuffle_size)
    ds = ds.batch(config.get("batch_size", 1))
    return ds

@tf.function
def load_wav(path):
    wav = tf.audio.decode_wav(tf.io.read_file(path))
    tf.debugging.assert_equal(wav.sample_rate, 16000, message="Failed to load audio file '{}'. Currently only 16 kHz sampling rate is supported.".format(path))
    # Merge channels by averaging over channels.
    # For mono this just drops the channel dim.
    return tf.math.reduce_mean(wav.audio, axis=1, keepdims=False)

@tf.function
def unbatch_ragged(ragged):
    return tf.data.Dataset.from_tensor_slices(ragged.to_tensor())

@tf.function
def not_empty(feats, meta):
    return not tf.reduce_all(tf.math.equal(feats, 0))

# Use batch_size > 1 iff _every_ audio file in paths has the same amount of samples
def extract_features(feat_config, paths, meta, batch_size=1):
    if "melspectrogram" in feat_config:
        feat_config["melspectrogram"]["sample_rate"] = feat_config["sample_rate"]
    for k in ("frame_length", "frame_step"):
        feat_config["voice_activity_detection"][k] = feat_config["spectrogram"][k]
    min_seq_len = feat_config["min_sequence_length"]
    not_too_short = lambda feats, meta: tf.math.greater(tf.size(feats), min_seq_len)
    extract_and_vad = lambda wavs: librosa_tf.extract_features_and_do_vad(
        wavs,
        feat_config["spectrogram"],
        feat_config["voice_activity_detection"],
        feat_config.get("melspectrogram"),
        feat_config.get("logmel"),
        feat_config.get("mfcc"),
    )
    paths = tf.constant(list(paths), dtype=tf.string)
    meta = tf.constant(list(meta), dtype=tf.string)
    tf.debugging.assert_equal(tf.shape(paths)[0], tf.shape(meta)[0], "The amount paths must match the length of the metadata list")
    features = (tf.data.Dataset
                  .from_tensor_slices(paths)
                  .map(load_wav, num_parallel_calls=AUTOTUNE)
                  .batch(batch_size)
                  .map(extract_and_vad, num_parallel_calls=AUTOTUNE)
                  .flat_map(unbatch_ragged))
    filtered = (tf.data.Dataset
                  .zip((features, tf.data.Dataset.from_tensor_slices(meta)))
                  .filter(not_empty)
                  .filter(not_too_short))
    feat_shape = (feat_config["feature_dim"],)
    global_min = filtered.reduce(
        tf.fill(feat_shape, float("inf")),
        lambda acc, x: tf.math.minimum(acc, tf.math.reduce_min(x[0], axis=0)))
    global_max = filtered.reduce(
        tf.fill(feat_shape, -float("inf")),
        lambda acc, x: tf.math.maximum(acc, tf.math.reduce_max(x[0], axis=0)))
    if "minmax_scaling" in feat_config:
        a = feat_config["minmax_scaling"]["min"]
        b = feat_config["minmax_scaling"]["max"]
        c = global_max - global_min
        scale_minmax = lambda feats, meta: (a + (b - a) * tf.math.divide_no_nan(feats - global_min, c), meta)
        filtered = filtered.map(scale_minmax)
    stats = {
        "global_min": global_min,
        "global_max": global_max,
    }
    return filtered, stats

def serialize_wav(wav, uuid, label):
    feature_definition = {
        "wav": floats2floatlist(wav),
        "uuid": string2byteslist(uuid),
        "label": string2byteslist(label),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_definition))
    return example.SerializeToString()

def deserialize_wav(example_str):
    feature_definition = {
        "wav": tf.io.VarLenFeature(dtype=tf.float32),
        "uuid": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }
    return tf.io.parse_single_example(example_str, feature_definition)

def write_wavs_to_tfrecord(wavs, target_path):
    with tf.io.TFRecordWriter(target_path, options=TFRECORD_COMPRESSION) as record_writer:
        for wav, meta in wavs:
            record_writer.write(serialize_wav(wav, **meta))
