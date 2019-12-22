import collections

from . import audio_feat
import kaldiio
import matplotlib.cm
import numpy as np
import tensorflow as tf


@tf.function
def feature_scaling(X, min, max, axis):
    """Apply feature scaling on X over given axis such that all values are between [min, max]"""
    X_min = tf.math.reduce_min(X, axis=axis, keepdims=True)
    X_max = tf.math.reduce_max(X, axis=axis, keepdims=True)
    return min + (max - min) * tf.math.divide_no_nan(X - X_min, X_max - X_min)

# TODO use variable length windows on the edges instead of zero padding, since zeros will dilute means and stddevs
@tf.function
def cmvn_slide(X, window_len=300):
    """Apply CMVN on batches of cepstral coef matrices X with a given cmvn window length."""
    tf.debugging.assert_rank_at_least(X, 3, message="Input to cmvn_slide should be batches of cepstral coef matrices (or tensors) with shape (Batch, Timedim, Coefs, ...)")
    # Pad beginning and end with zeros to fit window
    padding = tf.constant([[0, 0], [window_len//2, window_len//2 - 1 + (window_len&1)], [0, 0]])
    X_padded = tf.pad(X, padding, mode="CONSTANT", constant_values=0.0)
    cmvn_windows = tf.signal.frame(X_padded, window_len, 1, axis=1)
    tf.debugging.assert_equal(tf.shape(cmvn_windows)[1], tf.shape(X)[1], message="Mismatching amount of CMVN output windows and time steps in the input")
    # Standardize within each window and return result of same shape as X
    return tf.math.divide_no_nan(
        X - tf.math.reduce_mean(cmvn_windows, axis=2),
        tf.math.reduce_std(cmvn_windows, axis=2)
    )

@tf.function
def extract_features(signals, feattype, spec_kwargs, melspec_kwargs, mfcc_kwargs, db_spec_kwargs, feat_scale_kwargs, cmvn_kwargs):
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
    if feat_scale_kwargs:
        feat = feature_scaling(feat, **feat_scale_kwargs)
    if cmvn_kwargs:
        feat = cmvn_slide(feat, **cmvn_kwargs)
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
    if "frames" in feat_config:
        # Extract frames from all features, using the same metadata for each frame of one sample of features
        seq_len = feat_config["frames"]["length"]
        seq_step = feat_config["frames"]["step"]
        pad_zeros = feat_config["frames"].get("pad_zeros", False)
        if config.get("dataset_logger", {}).get("copy_original_audio", False):
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

def extract_features_for_prediction(feat_config, wav_config, paths, meta):
    paths = tf.constant(list(paths), dtype=tf.string)
    meta = tf.constant(list(meta), dtype=tf.string)
    tf.debugging.assert_equal(tf.shape(paths)[0], tf.shape(meta)[0], "The amount paths must match the length of the metadata list")
    wavs = tf.data.Dataset.from_tensor_slices((paths, meta)).map(load_wav, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if "wav_to_frames" in wav_config:
        frame_len = wav_config["wav_to_frames"]["length"]
        frame_step = wav_config["wav_to_frames"]["step"]
        pad_zeros = wav_config["wav_to_frames"].get("pad_zeros", False)
        wav_to_wavframes = lambda wav, *meta: (
            tf.signal.frame(wav, frame_len, frame_step, pad_end=pad_zeros, axis=0),
            *meta,
        )
        wavs = wavs.map(wav_to_wavframes).filter(not_empty)
    else:
        wavs = wavs.batch(1)
    extract_feats = lambda _wavs, *meta: (
        extract_features(
            _wavs,
            feat_config["type"],
            feat_config.get("spectrogram", {}),
            feat_config.get("melspectrogram", {}),
            feat_config.get("mfcc", {}),
            feat_config.get("db_spectrogram", {}),
            feat_config.get("sample_minmax_scaling", {}),
            feat_config.get("cmvn_kwargs", {}),
        ),
        *meta
    )
    features = wavs.map(extract_feats, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    min_seq_len = feat_config.get("min_sequence_length")
    if min_seq_len:
        def not_too_short(feats, *_):
            return tf.math.greater_equal(tf.shape(feats)[1], min_seq_len)
        features = features.filter(not_too_short)
    image_resize_kwargs = dict(feat_config.get("convert_to_images", {}))
    if image_resize_kwargs:
        # the 'size' key is always required
        image_size = image_resize_kwargs.pop("size")
        @tf.function
        def convert_to_images(feats, *meta):
            # Add single grayscale channel
            imgs = tf.expand_dims(feats, -1)
            # Ensure timesteps are the columns of the image
            imgs = tf.image.transpose(imgs)
            # Resize
            imgs = tf.image.resize(imgs, image_size, **image_resize_kwargs)
            return (imgs, *meta)
        features = features.map(convert_to_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if "frames" in feat_config:
        # Partition each utterance into frames without flattening to retain one tensor per utterance
        seq_len = feat_config["frames"]["length"]
        seq_step = feat_config["frames"]["step"]
        pad_zeros = feat_config["frames"].get("pad_zeros", False)
        feats_to_frames = lambda feats, *meta: (
            tf.signal.frame(feats, seq_len, seq_step, pad_end=pad_zeros, axis=1),
            *meta,
        )
        features = features.map(feats_to_frames)
    return features

def attach_dataset_logger(ds, features_name, max_image_samples=10, image_resize_kwargs=None, colormap="gray", copy_original_audio=False, debug_squeeze_last_dim=False):
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
        image, onehot, uttid = batch[:3]
        if debug_squeeze_last_dim:
            image = tf.squeeze(image, -1)
        # Scale grayscale channel between 0 and 1
        min, max = tf.math.reduce_min(image), tf.math.reduce_max(image)
        image = tf.math.divide_no_nan(image - min, max - min)
        # Map linear colormap over all grayscale values [0, 1] to produce an RGB image
        indices = tf.dtypes.cast(tf.math.round(image * num_colors), tf.int32)
        image = tf.gather(colors, indices)
        # Prepare for tensorboard
        if image_resize_kwargs:
            image = tf.image.transpose(image)
            image = tf.image.flip_up_down(image)
            image = tf.image.resize(image, **image_resize_kwargs)
        else:
            image = tf.image.flip_up_down(image)
        common = {"step": batch_idx, "max_outputs": max_image_samples}
        tf.summary.image(features_name, image, **common)
        if copy_original_audio:
            tf.summary.audio("original_audio", batch[3], 16000, **common)
        del common["max_outputs"]
        tf.summary.text("utterance_ids", uttid, **common)
        return batch
    return ds.enumerate().map(inspect_batches, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def without_metadata(dataset):
    return dataset.map(lambda feats, inputs, *meta: (feats, inputs))

@tf.function
def unbatch_ragged(ragged, meta):
    return tf.data.Dataset.from_tensor_slices((ragged.to_tensor(), meta))

@tf.function
def not_empty(x, *meta):
    return not (tf.size(x) == 0 or tf.reduce_all(x == 0))

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

def parse_sparsespeech_features(feat_config, enc_path, feat_path, seg2utt, utt2label):
    ss_input = kaldiio.load_scp(feat_path)
    with open(enc_path, "rb") as f:
        ss_encoding = np.load(f, fix_imports=False, allow_pickle=True).item()
    assert set(ss_input.keys()) == set(ss_encoding.keys()), "missing utterances, maybe all were not encoded?"
    keys = list(ss_encoding.keys())
    assert all(ss_encoding[k].ndim == ss_input[k].ndim for k in keys), "ss input and output must have the same dimensions"
    assert all(ss_encoding[k].shape[0] == ss_input[k].shape[0] for k in keys), "mismatching amount of time steps in ss input and the output encoding"
    encodingtype = tf.as_dtype(ss_encoding[keys[0]].dtype)
    noise_mean = feat_config.get("noise_mean", 0.0)
    noise_stddev = feat_config.get("noise_stddev", 0.01)
    feat_scale_kwargs = feat_config.get("sample_minmax_scaling", {})
    labels_only = feat_config.get("labels_only", False)
    def datagen():
        for seg_id, onehot_enc in ss_encoding.items():
            label = utt2label[seg2utt[seg_id]]
            noise = tf.random.normal(onehot_enc.shape, mean=noise_mean, stddev=noise_stddev, dtype=encodingtype)
            input_feat = ss_input[seg_id]
            output_feat = onehot_enc + noise
            # Apply feature scaling separately
            if feat_scale_kwargs:
                input_feat = feature_scaling(input_feat, **feat_scale_kwargs)
                output_feat = feature_scaling(output_feat, **feat_scale_kwargs)
            if labels_only:
                out = output_feat
            else:
                # Stack input and output features
                out = tf.concat((input_feat, output_feat), 1)
            yield out, (seg_id, label)
    return tf.data.Dataset.from_generator(
        datagen,
        (encodingtype, tf.string),
        (tf.TensorShape(feat_config["shape_after_concat"]), tf.TensorShape([2])),
    )

# Use batch_size > 1 iff _every_ audio file in paths has the same amount of samples
# TODO: fix this mess
def extract_features_from_paths(feat_config, wav_config, paths, meta, debug=False, copy_original_audio=False, trim_audio=None, debug_squeeze_last_dim=False):
    paths = tf.constant(list(paths), dtype=tf.string)
    meta = tf.constant(list(meta), dtype=tf.string)
    tf.debugging.assert_equal(tf.shape(paths)[0], tf.shape(meta)[0], "The amount paths must match the length of the metadata list")
    stats = collections.defaultdict(dict)
    wavs = (tf.data.Dataset.from_tensor_slices((paths, meta))
              .map(load_wav, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    # if debug:
        # stats = update_wav_summary(stats, wavs, "00_before_filtering")
    if "wav_to_frames" in wav_config:
        frame_len = wav_config["wav_to_frames"]["length"]
        frame_step = wav_config["wav_to_frames"]["step"]
        pad_zeros = wav_config["wav_to_frames"].get("pad_zeros", False)
        wav_to_wavframes = lambda wav, *meta: frame_and_unbatch(frame_len, frame_step, wav, meta, pad_zeros=pad_zeros)
        wavs = wavs.flat_map(wav_to_wavframes)
        # if debug:
            # stats = update_wav_summary(stats, wavs, "02_after_partitioning_to_frames")
    # This function expects batches of wavs
    extract_feats = lambda _wavs, *meta: (
        extract_features(
            _wavs,
            feat_config["type"],
            feat_config.get("spectrogram", {}),
            feat_config.get("melspectrogram", {}),
            feat_config.get("mfcc", {}),
            feat_config.get("db_spectrogram", {}),
            feat_config.get("sample_minmax_scaling", {}),
            feat_config.get("cmvn_kwargs", {}),
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
                  .map(extract_feats, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                  .unbatch())
    vad_config = dict(feat_config.get("voice_activity_detection", {}))
    if vad_config and vad_config.pop("match_feature_frames", False):
        _do_vad = lambda feats, meta, wav: (
            tf.boolean_mask(feats, tf.squeeze(audio_feat.framewise_energy_vad_decisions(tf.expand_dims(wav, 0), **vad_config), 0)),
            meta,
            wav,
        )
        features = features.map(_do_vad)
    min_seq_len = feat_config.get("min_sequence_length")
    if min_seq_len:
        def not_too_short(feats, *_):
            return tf.math.greater_equal(tf.shape(feats)[0], min_seq_len)
        features = features.filter(not_too_short)
        # if debug:
            # stats = update_feat_summary(stats, features, "01_after_too_short_filter")
    image_resize_kwargs = dict(feat_config.get("convert_to_images", {}))
    if image_resize_kwargs:
        # the 'size' key is always required
        image_size = image_resize_kwargs.pop("size")
        @tf.function
        def convert_to_images(feats, *meta):
            # Add single grayscale channel
            imgs = tf.expand_dims(feats, -1)
            # Ensure timesteps are the columns of the image
            imgs = tf.image.transpose(imgs)
            # Resize
            imgs = tf.image.resize(imgs, image_size, **image_resize_kwargs)
            return (imgs, *meta)
        features = features.map(convert_to_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # if debug:
        # stats = update_feat_summary(stats, features, "00_before_filtering")
    # global_scale_conf = feat_config.get("global_minmax_scaling")
    # if global_scale_conf:
    #     feat_shape = (feat_config["feature_dim"],)
    #     if debug_squeeze_last_dim:
    #         without_channels = features.map(lambda feats, *meta: (tf.squeeze(feats, -1), *meta))
    #     else:
    #         without_channels = features
    #     stats["features"]["global_min"] = reduce_min(without_channels, shape=feat_shape)
    #     stats["features"]["global_max"] = reduce_max(without_channels, shape=feat_shape)
    #     # Apply feature scaling on each feature dimension over whole dataset
    #     a = global_scale_conf["min"]
    #     b = global_scale_conf["max"]
    #     c = stats["features"]["global_max"] - stats["features"]["global_min"]
    #     scale_minmax = lambda feats, *meta: (a + (b - a) * tf.math.divide_no_nan(feats - global_min, c), *meta)
    #     features = features.map(scale_minmax)
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
