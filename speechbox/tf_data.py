from multiprocessing import cpu_count

from . import librosa_tf
import numpy as np
import tensorflow as tf


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

# TODO this is horribly slow
def write_features(extractor_dataset, target_path):
    if not target_path.endswith(".tfrecord"):
        target_path += ".tfrecord"
    serialize = lambda feats, meta: tf.py_function(serialize_sequence_features, (feats, meta), tf.string)
    record_writer = tf.data.experimental.TFRecordWriter(target_path, compression_type=TFRECORD_COMPRESSION)
    record_writer.write(extractor_dataset.map(serialize))

def load_features(tfrecord_paths, feature_dim, dataset_config):
    deserialize = lambda s: deserialize_sequence_features(s, feature_dim)
    ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type=TFRECORD_COMPRESSION)
    return ds.map(deserialize)

@tf.function
def frame_and_unbatch(sequences, meta, frame_len, frame_step):
    frames = tf.signal.frame(sequences, frame_len, frame_step, axis=0)
    # Repeat same meta for all frames
    frames_meta = tf.tile(tf.expand_dims(meta, 0), [tf.shape(frames)[0], 1])
    # Zip together
    return tf.data.Dataset.from_tensor_slices((frames, frames_meta))

def prepare_dataset_for_training(ds, config, label2onehot):
    if "frames" in config:
        # Extract frames from all features, using the same metadata for each frame of one sample of features
        seq_len, seq_step = config["frames"]["length"], config["frames"]["step"]
        ds = ds.flat_map(lambda feats, meta: frame_and_unbatch(feats, meta, seq_len, seq_step))
    image_size = config.get("resize_input_as_images")
    if image_size:
        def convert_to_images(feats, meta):
            # Add single grayscale channel
            imgs = tf.expand_dims(feats, -1)
            # Resize
            imgs = tf.image.resize(imgs, (image_size["width"], image_size["height"]))
            return imgs, meta
        ds = ds.map(convert_to_images)
    to_model_input = lambda feats, meta: (feats, label2onehot(meta[1]), meta[0])
    ds = ds.map(to_model_input)
    if "shuffle_buffer_size" in config:
        ds = ds.shuffle(config["shuffle_buffer_size"])
    ds = ds.batch(config.get("batch_size", 1))
    if "prefetch" in config:
        ds = ds.prefetch(config["prefetch"])
    return ds

def attach_dataset_logger(ds, max_image_samples=10, image_size=None, expand_channel_dim=False):
    @tf.function
    def inspect_batches(batch_idx, batch):
        image, onehot, meta = batch
        if expand_channel_dim:
            # 'image' is not actually an image yet, add grayscale channel
            image = tf.expand_dims(image, -1)
        # Scale grayscale channel between 0 and 1
        min, max = tf.math.reduce_min(image), tf.math.reduce_max(image)
        image = tf.math.divide_no_nan(image - min, max - min)
        image = tf.image.transpose(image)
        image = tf.image.flip_up_down(image)
        if image_size:
            image = tf.image.resize_with_pad(image, *image_size, method="nearest")
        tf.summary.image("features", image, step=batch_idx, max_outputs=max_image_samples)
        return batch
    ds = ds.enumerate().map(inspect_batches)
    return ds

def without_metadata(dataset):
    return dataset.map(lambda feats, inputs, *meta: (feats, inputs))

@tf.function
def load_wav(path, meta):
    wav = tf.audio.decode_wav(tf.io.read_file(path))
    tf.debugging.assert_equal(wav.sample_rate, 16000, message="Failed to load audio file '{}'. Currently only 16 kHz sampling rate is supported.".format(path))
    # Merge channels by averaging over channels.
    # For mono this just drops the channel dim.
    return (tf.math.reduce_mean(wav.audio, axis=1, keepdims=False), meta)

@tf.function
def unbatch_ragged(ragged, meta):
    return tf.data.Dataset.from_tensor_slices((ragged.to_tensor(), meta))

@tf.function
def not_empty(feats, meta):
    return not tf.reduce_all(tf.math.equal(feats, 0))

# Use batch_size > 1 iff _every_ audio file in paths has the same amount of samples
def extract_features(feat_config, paths, meta, num_parallel_calls=cpu_count()):
    paths = tf.constant(list(paths), dtype=tf.string)
    meta = tf.constant(list(meta), dtype=tf.string)
    tf.debugging.assert_equal(tf.shape(paths)[0], tf.shape(meta)[0], "The amount paths must match the length of the metadata list")
    if feat_config["type"] == "sparsespeech":
        with open(feat_config["path"], "rb") as f:
            data = np.load(f, fix_imports=False, allow_pickle=True).item()
        data = [(data[m[0].numpy().decode("utf-8")], m) for m in meta]
        datatype = tf.as_dtype(data[0][0].dtype)
        def datagen():
            for d, m in data:
                noise = tf.random.normal(d.shape, mean=0, stddev=0.01, dtype=datatype)
                yield d + noise, m
        features = tf.data.Dataset.from_generator(
            datagen,
            (datatype, tf.string),
            (tf.TensorShape([None, feat_config["feature_dim"]]), tf.TensorShape([2])),
        )
    else:
        extract_and_vad = lambda wavs, meta: (
            librosa_tf.extract_features_and_do_vad(
                wavs,
                feat_config["type"],
                feat_config.get("spectrogram", {}),
                feat_config.get("voice_activity_detection", {}),
                feat_config.get("melspectrogram", {}),
                feat_config.get("mfcc", {}),
            ),
            meta
        )
        wavs = (tf.data.Dataset.from_tensor_slices((paths, meta))
                  .map(load_wav, num_parallel_calls=num_parallel_calls))
        if "frames" in feat_config:
            frame_len, frame_step = feat_config["frames"]["length"], feat_config["frames"]["step"]
            wavs  = wavs.flat_map(lambda wavs, meta: frame_and_unbatch(wavs, meta, frame_len, frame_step))
        features = (wavs
                      .batch(feat_config.get("batch_size", 1))
                      .map(extract_and_vad, num_parallel_calls=num_parallel_calls)
                      .flat_map(unbatch_ragged)
                      .filter(not_empty))
    min_seq_len = feat_config.get("min_sequence_length", 0)
    not_too_short = lambda feats, meta: tf.math.greater(tf.size(feats), min_seq_len)
    features = features.filter(not_too_short)
    feat_shape = (feat_config["feature_dim"],)
    global_min = features.reduce(
        tf.fill(feat_shape, float("inf")),
        lambda acc, x: tf.math.minimum(acc, tf.math.reduce_min(x[0], axis=0)))
    global_max = features.reduce(
        tf.fill(feat_shape, -float("inf")),
        lambda acc, x: tf.math.maximum(acc, tf.math.reduce_max(x[0], axis=0)))
    scale_conf = feat_config.get("global_minmax_scaling")
    if scale_conf:
        # Apply feature scaling on each feature dimension over whole dataset
        a = scale_conf["min"]
        b = scale_conf["max"]
        c = global_max - global_min
        scale_minmax = lambda feats, meta: (a + (b - a) * tf.math.divide_no_nan(feats - global_min, c), meta)
        features = features.map(scale_minmax)
    stats = {
        "global_min": global_min,
        "global_max": global_max,
    }
    return features, stats

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
