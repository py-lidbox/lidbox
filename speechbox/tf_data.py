from multiprocessing import cpu_count
import collections

from . import audio_feat
import matplotlib.cm
import numpy as np
import tensorflow as tf


@tf.function
def extract_features(signals, feattype, spec_kwargs, melspec_kwargs, mfcc_kwargs, db_spec_kwargs):
    feat = audio_feat.spectrograms(signals, **spec_kwargs)
    if feattype in ("melspectrogram", "logmelspectrogram", "mfcc"):
        feat = audio_feat.melspectrograms(feat, **melspec_kwargs)
        if feattype in ("logmelspectrogram", "mfcc"):
            feat = tf.math.log(feat + 1e-6)
            if feattype == "mfcc":
                coef_begin = mfcc_kwargs.get("coef_begin", 1)
                coef_end = mfcc_kwargs.get("coef_end", 13)
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(feat)
                feat = mfccs[..., coef_begin:coef_end]
    elif feattype in ("db_spectrogram",):
        feat = audio_feat.power_to_db(feat, **db_spec_kwargs)
    return feat

@tf.function
def load_wav(path, meta):
    wav = tf.audio.decode_wav(tf.io.read_file(path))
    tf.debugging.assert_equal(wav.sample_rate, 16000, message="Failed to load audio file '{}'. Currently only 16 kHz sampling rate is supported.".format(path))
    # Merge channels by averaging over channels.
    # For mono this just drops the channel dim.
    return (tf.math.reduce_mean(wav.audio, axis=1, keepdims=False), meta)

@tf.function
def write_wav(path, wav):
    tf.debugging.assert_rank(wav, 2, "write_wav expects signals with shape [N, c] where N is amount of samples and c channels.")
    return tf.io.write_file(path, tf.audio.encode_wav(wav, 16000))

@tf.function
def count_dataset(ds):
    return ds.reduce(tf.constant(0, dtype=tf.int64), lambda c, *_: c + 1)

@tf.function
def reduce_mean(ds, shape=[]):
    sum = ds.reduce(
        tf.zeros(shape, dtype=tf.float64),
        lambda sum, t: sum + tf.dtypes.cast(t[0], tf.float64)
    )
    return sum / tf.dtypes.cast(count_dataset(ds), tf.float64)

@tf.function
def reduce_min(ds, shape=[]):
    return ds.reduce(
        tf.fill(shape, tf.dtypes.cast(float("inf"), tf.float64)),
        lambda min, t: tf.math.minimum(min, tf.math.reduce_min(tf.dtypes.cast(t[0], tf.float64), axis=0))
    )

@tf.function
def reduce_max(ds, shape=[]):
    return ds.reduce(
        tf.fill(shape, tf.dtypes.cast(-float("inf"), tf.float64)),
        lambda max, t: tf.math.maximum(max, tf.math.reduce_max(tf.dtypes.cast(t[0], tf.float64), axis=0))
    )

@tf.function
def frame_and_unbatch(frame_len, frame_step, features, meta, pad_zeros=False):
    frames = tf.signal.frame(features, frame_len, frame_step, pad_end=pad_zeros, axis=0)
    # Skip dummy wavs
    meta = (meta[0], *meta[2:])
    # Repeat the same context meta over all frames (subsequences)
    frames_with_meta = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(frames),
        *[tf.data.Dataset.from_tensors(m).repeat() for m in meta],
    ))
    return frames_with_meta

@tf.function
def frame_and_unbatch_with_wav_in_meta(frame_len, frame_step, features, meta, feat_frame_len, feat_frame_step, pad_zeros=False):
    frames = tf.signal.frame(features, frame_len, frame_step, pad_end=pad_zeros, axis=0)
    # Take frames of the waveform such that each frame match the frames taken from 'features'
    wav_window_len = feat_frame_len + (frame_len-1)*feat_frame_step
    wav_window_step = feat_frame_len + (frame_step-1)*feat_frame_step
    meta_frames = tf.signal.frame(meta[1], wav_window_len, wav_window_step, pad_end=pad_zeros)
    frames_with_meta = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(frames),
        tf.data.Dataset.from_tensors(meta[0]).repeat(),
        tf.data.Dataset.from_tensor_slices(meta_frames),
        *[tf.data.Dataset.from_tensors(m).repeat() for m in meta[2:]],
    ))
    return frames_with_meta

def prepare_dataset_for_training(ds, config, feat_config, label2onehot):
    if "frames" in config:
        # Extract frames from all features, using the same metadata for each frame of one sample of features
        seq_len, seq_step = config["frames"]["length"], config["frames"]["step"]
        pad_zeros = config["frames"].get("pad_zeros", False)
        if "dataset_logger" in config:
            # Original waveforms have been copied into the metadata and we need to split those to frames also
            to_frames = lambda feats, *meta: frame_and_unbatch_with_wav_in_meta(
                seq_len,
                seq_step,
                feats,
                meta,
                feat_config["spectrogram"]["frame_length"],
                feat_config["spectrogram"]["frame_step"],
                pad_zeros=pad_zeros
            )
        else:
            # index 1 in meta are dummy wavs with length 0
            to_frames = lambda feats, *meta: frame_and_unbatch(
                seq_len,
                seq_step,
                feats,
                meta,
                pad_zeros=pad_zeros
            )
        ds = ds.flat_map(to_frames)
    # Transform dataset such that 2 first elements will always be (sample, onehot_label) and rest will be metadata that can be safely dropped when training starts
    def to_model_input(feats, *meta):
        model_input = (
            feats,
            label2onehot(meta[0][1]),
            # utterance id
            meta[0][0],
        )
        if len(meta) == 1:
            return model_input
        else:
            # original waveform, reshaping the array here to add a mono channel
            return model_input + (tf.expand_dims(meta[1], -1), *meta[2:])
    ds = ds.map(to_model_input)
    shuffle_buffer_size = config.get("shuffle_buffer_size", 0)
    if shuffle_buffer_size:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(config.get("batch_size", 1))
    if "prefetch" in config:
        ds = ds.prefetch(config["prefetch"])
    return ds

def attach_dataset_logger(ds, features_name, max_image_samples=10, image_size=None, colormap="gray", copy_original_audio=False, debug_squeeze_last_dim=False):
    """
    Write Tensorboard summary information for samples in the given tf.data.Dataset.
    """
    # TF colormap trickery from https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
    # The idea is to extract all RGB values from the matplotlib colormap into a tf.constant
    cmap = matplotlib.cm.get_cmap(colormap)
    num_colors = cmap.N
    colors = tf.constant(cmap(np.arange(num_colors + 1))[:,:3], dtype=tf.float32)
    @tf.function
    def inspect_batches(batch_idx, batch):
        image, onehot, uttid, signal = batch
        if debug_squeeze_last_dim:
            image = tf.squeeze(image, -1)
        # Scale grayscale channel between 0 and 1
        min, max = tf.math.reduce_min(image), tf.math.reduce_max(image)
        image = tf.math.divide_no_nan(image - min, max - min)
        # Map linear colormap over all grayscale values [0, 1] to produce an RGB image
        indices = tf.dtypes.cast(tf.math.round(image * num_colors), tf.int32)
        image = tf.gather(colors, indices)
        # Prepare for tensorboard
        image = tf.image.transpose(image)
        image = tf.image.flip_up_down(image)
        if image_size:
            image = tf.image.resize(image, image_size, method="nearest")
        common = {"step": batch_idx, "max_outputs": max_image_samples}
        tf.summary.image(features_name, image, **common)
        if copy_original_audio:
            tf.summary.audio("original_audio", signal, 16000, **common)
        del common["max_outputs"]
        tf.summary.text("utterance_ids", uttid, **common)
        return batch
    return ds.enumerate().map(inspect_batches)

def without_metadata(dataset):
    return dataset.map(lambda feats, inputs, *meta: (feats, inputs))

@tf.function
def unbatch_ragged(ragged, meta):
    return tf.data.Dataset.from_tensor_slices((ragged.to_tensor(), meta))

@tf.function
def not_empty(feats, meta):
    return not tf.reduce_all(tf.math.equal(feats, 0))

def update_wav_summary(stats, wav_ds, key):
    stats["num_wavs"][key] = count_dataset(wav_ds)
    wav_sizes = wav_ds.map(lambda wav, *meta: (tf.expand_dims(tf.dtypes.cast(tf.size(wav), tf.float64), -1), *meta))
    stats["mean_wav_length"][key] = reduce_mean(wav_sizes)
    stats["min_wav_length"][key] = reduce_min(wav_sizes)
    stats["max_wav_length"][key] = reduce_max(wav_sizes)
    return stats

def update_feat_summary(stats, feat_ds, key):
    stats["num_feats"][key] = count_dataset(feat_ds)
    feat_num_frames = feat_ds.map(lambda feat, *meta: (tf.expand_dims(tf.dtypes.cast(tf.shape(feat)[0], tf.float64), -1), *meta))
    stats["mean_feat_length"][key] = reduce_mean(feat_num_frames)
    stats["min_feat_length"][key] = reduce_min(feat_num_frames)
    stats["max_feat_length"][key] = reduce_max(feat_num_frames)
    return stats

# Use batch_size > 1 iff _every_ audio file in paths has the same amount of samples
# TODO: fix this mess
def extract_features_from_paths(feat_config, paths, meta, debug=False, copy_original_audio=False, trim_audio=None, debug_squeeze_last_dim=False):
    paths = tf.constant(list(paths), dtype=tf.string)
    meta = tf.constant(list(meta), dtype=tf.string)
    tf.debugging.assert_equal(tf.shape(paths)[0], tf.shape(meta)[0], "The amount paths must match the length of the metadata list")
    stats = collections.defaultdict(dict)
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
        wavs = (tf.data.Dataset.from_tensor_slices((paths, meta))
                  .map(load_wav, num_parallel_calls=8*cpu_count()))
        if debug:
            stats = update_wav_summary(stats, wavs, "00_before_filtering")
        vad_config = feat_config.get("voice_activity_detection")
        if vad_config:
            wavs = wavs.map(lambda wav, *meta: (audio_feat.energy_vad(wav, **vad_config), *meta))
            if debug:
                stats = update_wav_summary(stats, wavs, "01_after_vad_filter")
        if "frames" in feat_config:
            frame_len, frame_step = feat_config["frames"]["length"], feat_config["frames"]["step"]
            pad_zeros = feat_config["frames"].get("pad_zeros", False)
            wavs = wavs.flat_map(lambda wav, *meta: frame_and_unbatch(frame_len, frame_step, wav, meta, pad_zeros=pad_zeros))
            if debug:
                stats = update_wav_summary(stats, wavs, "02_after_partitioning_to_frames")
        # This function expects batches of wavs
        extract_feats = lambda wavs, *meta: (
            extract_features(
                wavs,
                feat_config["type"],
                feat_config.get("spectrogram", {}),
                feat_config.get("melspectrogram", {}),
                feat_config.get("mfcc", {}),
                feat_config.get("db_spectrogram", {}),
            ),
            *meta
        )
        # Duplicate audio data into meta so we can listen to it in Tensorboard
        # Feature extraction will replace the non-meta audio data with features
        wavs_extended = tf.data.Dataset.zip((
            wavs.map(lambda wav, _: wav),
            # We expect the metadata to still be in a single tensor
            wavs.map(lambda _, meta: meta),
            # Copy the waveform into new tensor or use a dummy wav with a single zero frame
            wavs.map(lambda wav, _: tf.identity(wav) if copy_original_audio else tf.zeros([1])),
        ))
        if trim_audio:
            # Ensure all wav files in the metadata contains precisely 'trim_audio' amount of frames
            def pad_or_slice_metawavs(wav, meta, metawav):
                padding = [[0, tf.maximum(0, trim_audio - tf.size(metawav))]]
                trimmed = tf.pad(metawav[:trim_audio], padding)
                return (wav, meta, trimmed)
            wavs_extended = wavs_extended.map(pad_or_slice_metawavs)
        features = (wavs_extended
                      .batch(feat_config.get("batch_size", 1))
                      .map(extract_feats, num_parallel_calls=2*cpu_count())
                      .unbatch())
    min_seq_len = feat_config.get("min_sequence_length")
    if min_seq_len:
        def not_too_short(feats, *_):
            return tf.math.greater_equal(tf.shape(feats)[0], min_seq_len)
        features = features.filter(not_too_short)
        if debug:
            stats = update_feat_summary(stats, features, "01_after_too_short_filter")
    image_size = feat_config.get("resize_as_image")
    if image_size:
        def convert_to_images(feats, *meta):
            # Add single grayscale channel
            imgs = tf.expand_dims(feats, -1)
            # Resize
            imgs = tf.image.resize(imgs, (image_size["width"], image_size["height"]))
            return (imgs, *meta)
        features = features.map(convert_to_images)
    if debug:
        stats = update_feat_summary(stats, features, "00_before_filtering")
    sample_scale_conf = feat_config.get("sample_minmax_scaling")
    if sample_scale_conf:
        # Apply feature scaling on each sample
        a = sample_scale_conf["min"]
        b = sample_scale_conf["max"]
        axis = sample_scale_conf["axis"]
        @tf.function
        def scale_minmax(feats, *meta):
            min = tf.math.reduce_min(feats, axis=axis, keepdims=True)
            max = tf.math.reduce_max(feats, axis=axis, keepdims=True)
            return (a + (b - a) * tf.math.divide_no_nan(feats - min, max - min), *meta)
        features = features.map(scale_minmax)
    global_scale_conf = feat_config.get("global_minmax_scaling")
    if debug or global_scale_conf:
        feat_shape = (feat_config["feature_dim"],)
        if debug_squeeze_last_dim:
            without_channels = features.map(lambda feats, *meta: (tf.squeeze(feats, -1), *meta))
        else:
            without_channels = features
        stats["features"]["global_min"] = reduce_min(without_channels, shape=feat_shape)
        stats["features"]["global_max"] = reduce_max(without_channels, shape=feat_shape)
    if global_scale_conf:
        # Apply feature scaling on each feature dimension over whole dataset
        a = global_scale_conf["min"]
        b = global_scale_conf["max"]
        c = stats["features"]["global_max"] - stats["features"]["global_min"]
        scale_minmax = lambda feats, *meta: (a + (b - a) * tf.math.divide_no_nan(feats - global_min, c), *meta)
        features = features.map(scale_minmax)
    return features, stats


# TF serialization functions, not really needed if features are cached using tf.data.Dataset.cache

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
