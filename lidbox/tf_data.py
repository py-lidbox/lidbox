import collections
import contextlib
import io
import json
import os
import random
import shutil
import sys
import time
import wave

import kaldiio
import librosa.core
import matplotlib.cm
import numpy as np
import soundfile
import tensorflow as tf
import webrtcvad

import lidbox
if lidbox.TF_DEBUG:
    tf.autograph.set_verbosity(10, alsologtostdout=True)
    TF_AUTOTUNE = None
else:
    TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

from . import (
    audio_feat,
    tf_util,
    yaml_pprint,
)

@tf.function
def feature_scaling(X, min, max, axis=None):
    """Apply feature scaling on X over given axis such that all values are between [min, max]"""
    X_min = tf.math.reduce_min(X, axis=axis, keepdims=True)
    X_max = tf.math.reduce_max(X, axis=axis, keepdims=True)
    return min + (max - min) * tf.math.divide_no_nan(X - X_min, X_max - X_min)

@tf.function
def mean_var_norm_slide(X, window_len=300, normalize_variance=True):
    """Apply mean and variance norm on batches of features matrices X with a given window length."""
    tf.debugging.assert_rank(X, 3, message="Input to mean_var_norm_slide should be of shape (batch_size, timedim, channels)")
    if tf.shape(X)[1] <= window_len:
        # All frames of X fit inside one window, no need for sliding window
        centered = X - tf.math.reduce_mean(X, axis=1, keepdims=True)
        if normalize_variance:
            return tf.math.divide_no_nan(centered, tf.math.reduce_std(X, axis=1, keepdims=True))
        else:
            return centered
    else:
        # Padding by reflecting the coefs along the time dimension should not dilute the means and variances as much as zeros would
        padding = tf.constant([[0, 0], [window_len//2, window_len//2 - 1 + (window_len&1)], [0, 0]])
        X_padded = tf.pad(X, padding, mode="REFLECT")
        windows = tf.signal.frame(X_padded, window_len, 1, axis=1)
        tf.debugging.assert_equal(tf.shape(windows)[1], tf.shape(X)[1], message="Mismatching amount of output windows and time steps in the input")
        centered = X - tf.math.reduce_mean(windows, axis=2)
        if normalize_variance:
            return tf.math.divide_no_nan(centered, tf.math.reduce_std(windows, axis=2))
        else:
            return centered

# NOTE tensorflow does not yet support non-zero axes in tf.gather when indices are ragged
# @tf.function
# def mean_var_norm_gather(X, window_len=300):
#     """Same as mean_var_norm_slide but without padding."""
#     tf.debugging.assert_rank_at_least(X, 3)
#     num_total_frames = tf.shape(X)[1]
#     begin = tf.range(0, num_total_frames) - window_len // 2 + 1
#     end = begin + window_len
#     begin = tf.clip_by_value(begin, 0, num_total_frames)
#     end = tf.clip_by_value(end, 0, num_total_frames)
#     window_indices = tf.ragged.range(begin, end)
#     windows = tf.gather(X, window_indices, axis=1)
#     return tf.math.divide_no_nan(
#         X - tf.math.reduce_mean(windows, axis=2),
#         tf.math.reduce_std(windows, axis=2)
#     )

def mean_var_norm_nopad_slide_numpy(X, window_len, normalize_variance):
    num_total_frames = X.shape[1]
    if num_total_frames <= window_len:
        centered = X - np.mean(X, axis=1, keepdims=True)
        if normalize_variance:
            centered /= np.std(X, axis=1, keepdims=True)
        return centered
    begin = np.arange(0, num_total_frames) - window_len // 2
    end = begin + window_len
    begin = np.clip(begin, 0, num_total_frames)
    end = np.clip(end, 0, num_total_frames)
    result = np.zeros_like(X)
    for i, (b, e) in enumerate(zip(begin, end)):
        window = X[:,b:e]
        centered = X[:,i] - np.mean(window, axis=1)
        if normalize_variance:
            centered /= np.std(window, axis=1)
        result[:,i] = centered
    return result

@tf.function
def extract_features(signals, feattype, spec_kwargs, melspec_kwargs, mfcc_kwargs, db_spec_kwargs, feat_scale_kwargs, mean_var_norm_kwargs):
    sample_rate = signals.sample_rate[0]
    tf.debugging.assert_equal(signals.sample_rate, [sample_rate], message="All signals in the feature extraction batch must have equal sample rates")
    feat = audio_feat.spectrograms(signals, **spec_kwargs)
    tf.debugging.assert_all_finite(feat, "spectrogram failed")
    if feattype in ("melspectrogram", "logmelspectrogram", "mfcc"):
        feat = audio_feat.melspectrograms(feat, sample_rate=sample_rate, **melspec_kwargs)
        tf.debugging.assert_all_finite(feat, "melspectrogram failed")
        if feattype in ("logmelspectrogram", "mfcc"):
            feat = tf.math.log(feat + 1e-6)
            tf.debugging.assert_all_finite(feat, "logmelspectrogram failed")
            if feattype == "mfcc":
                coef_begin = mfcc_kwargs.get("coef_begin", 1)
                coef_end = mfcc_kwargs.get("coef_end", 13)
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(feat)
                feat = mfccs[..., coef_begin:coef_end]
                tf.debugging.assert_all_finite(feat, "mfcc failed")
    elif feattype in ("db_spectrogram",):
        feat = audio_feat.power_to_db(feat, **db_spec_kwargs)
        tf.debugging.assert_all_finite(feat, "db_spectrogram failed")
    if feat_scale_kwargs:
        feat = feature_scaling(feat, **feat_scale_kwargs)
        tf.debugging.assert_all_finite(feat, "feature scaling failed")
    if mean_var_norm_kwargs:
        feat = mean_var_norm_slide(feat, **mean_var_norm_kwargs)
        tf.debugging.assert_all_finite(feat, "mean_var_norm failed")
    return feat

def feat_extraction_args_as_list(feat_config):
    args = [feat_config["type"]]
    kwarg_dicts = [
        feat_config.get("spectrogram", {}),
        feat_config.get("melspectrogram", {}),
        feat_config.get("mfcc", {}),
        feat_config.get("db_spectrogram", {}),
        feat_config.get("sample_minmax_scaling", {}),
        feat_config.get("mean_var_norm_slide", {}),
    ]
    # all dict values could be converted explicitly to tensors here if tf.functions are behaving badly
    # args.extend([tf_convert(d) for d in kwarg_dicts])
    return args + kwarg_dicts

def count_dataset(ds):
    return ds.reduce(tf.constant(0, tf.int64), lambda c, elem: c + 1)

def filter_with_min_shape(ds, min_shape, ds_index=0):
    min_shape = tf.constant(min_shape, dtype=tf.int32)
    at_least_min_shape = lambda *t: tf.math.reduce_all(tf.shape(t[ds_index]) >= min_shape)
    return ds.filter(at_least_min_shape)

# def append_webrtcvad_decisions(ds, config):
#     assert 0 <= config["aggressiveness"] <= 3, "webrtcvad aggressiveness values must be in range [0, 3]"
#     # TODO assert encoded frame lengths are in {10, 20, 30} ms since webrtcvad supports only those
#     aggressiveness = tf.constant(config["aggressiveness"], tf.int32)
#     vad_frame_length = tf.constant(config["vad_frame_length"], tf.int32)
#     vad_frame_step = tf.constant(config["vad_frame_step"], tf.int32)
#     feat_frame_length = tf.constant(config["feat_frame_length"], tf.int32)
#     feat_frame_step = tf.constant(config["feat_frame_step"], tf.int32)
#     wavs_to_pcm_data = lambda wav, *rest: (wav, audio_feat.wav_to_pcm_data(wav)[1], *rest)
#     apply_webrtcvad = lambda wav, wav_bytes, *rest: (
#         wav,
#         *rest,
#         tf.numpy_function(
#             audio_feat.framewise_webrtcvad_decisions,
#             [tf.size(wav.audio),
#              wav_bytes,
#              wav.sample_rate,
#              vad_frame_length,
#              vad_frame_step,
#              feat_frame_length,
#              feat_frame_step,
#              aggressiveness],
#             tf.bool))
#     return (ds.map(wavs_to_pcm_data, num_parallel_calls=TF_AUTOTUNE)
#               .map(apply_webrtcvad, num_parallel_calls=TF_AUTOTUNE))

def append_mfcc_energy_vad_decisions(ds, config):
    vad_kwargs = dict(config)
    spec_kwargs = vad_kwargs.pop("spectrogram")
    melspec_kwargs = vad_kwargs.pop("melspectrogram")
    f = lambda wav, *rest: (
        tf_util.tf_print("mfcc vad", rest[0]) and wav,
        *rest,
        audio_feat.framewise_mfcc_energy_vad_decisions(wav, spec_kwargs, melspec_kwargs, **vad_kwargs))
    return ds.map(f, num_parallel_calls=TF_AUTOTUNE)

def make_random_frame_chunker_fn(len_config):
    float_lengths = tf.linspace(
        float(len_config["min"]),
        float(len_config["max"]),
        int(len_config["num_bins"]))
    lengths = tf.unique(tf.cast(float_lengths, tf.int32))[0]
    min_chunk_length = lengths[0]
    tf.debugging.assert_greater(min_chunk_length, 1, message="Too short minimum chunk length")
    min_overlap = tf.constant(float(len_config.get("min_overlap", 0)), tf.float32)
    tf.debugging.assert_less(min_overlap, 1.0, message="Minimum overlap ratio of two adjacent random chunks must be less than 1.0")
    @tf.function
    def chunk_timedim_randomly(features):
        num_total_frames = tf.shape(features)[0]
        max_num_chunks = 1 + tf.math.maximum(0, num_total_frames - min_chunk_length)
        rand_length_indexes = tf.random.uniform([max_num_chunks], 0, tf.size(lengths), dtype=tf.int32)
        rand_chunk_lengths = tf.gather(lengths, rand_length_indexes)
        rand_offset_ratios = tf.random.uniform([tf.size(rand_chunk_lengths)], 0.0, 1.0 - min_overlap, dtype=tf.float32)
        offsets = tf.cast(rand_offset_ratios * tf.cast(rand_chunk_lengths, tf.float32), tf.int32)
        offsets = tf.concat(([0], tf.math.maximum(1, offsets[:-1])), axis=0)
        begin = tf.math.cumsum(offsets)
        begin = tf.boolean_mask(begin, begin < num_total_frames)
        end = begin + tf.boolean_mask(rand_chunk_lengths, begin < num_total_frames)
        end = tf.math.minimum(num_total_frames, end)
        # TODO gather is overkill here since we only want several slices
        chunk_indices = tf.ragged.range(begin, end)
        return tf.gather(features, chunk_indices)
    return chunk_timedim_randomly

def group_by_sequence_length(ds, min_batch_size, max_batch_size, verbosity=0, sequence_dim=0):
    if verbosity:
        tf_util.tf_print("Grouping samples by sequence length into batches of max size", max_batch_size)
    def get_seq_len(feat, *args):
        return tf.cast(tf.shape(feat)[sequence_dim], tf.int64)
    def group_to_batch(key, group, *args):
        return group.batch(max_batch_size)
    ds = ds.apply(tf.data.experimental.group_by_window(get_seq_len, group_to_batch, window_size=max_batch_size))
    if min_batch_size:
        if verbosity:
            tf_util.tf_print("Dropping batches smaller than min_batch_size", min_batch_size)
        min_batch_size = tf.constant(min_batch_size, tf.int32)
        has_min_batch_size = lambda batch, *args: (tf.shape(batch)[0] >= min_batch_size)
        ds = ds.filter(has_min_batch_size)
    return ds

def prepare_dataset_for_training(ds, config, feat_config, label2onehot, model_id, conf_checksum='', verbosity=0):
    if "frames" in config or "random_frames" in config:
        if "frames" in config:
            frame_config = config["frames"]
            if verbosity:
                print("Dividing the time axis of features into fixed length chunks")
                if verbosity > 1:
                    print("Using fixed length config:")
                    yaml_pprint(frame_config)
            # Extract frames from all features, using the same metadata for each frame of one sample of features
            def to_frames(feats, *meta):
                feats_in_frames = tf.signal.frame(
                        feats,
                        frame_config["length"],
                        frame_config["step"],
                        pad_end=frame_config.get("pad_zeros", False),
                        axis=0)
                return (feats_in_frames, *meta)
            ds = ds.map(to_frames, num_parallel_calls=TF_AUTOTUNE)
        elif "random_frames" in config:
            frame_config = config["random_frames"]
            if verbosity:
                print("Dividing the time axis of features into random length chunks")
                if verbosity > 1:
                    print("Using random chunk config:")
                    yaml_pprint(frame_config)
            if "cache_prepared_dataset_to_tmpdir" not in config:
                print("Warning: 'cache_prepared_dataset_to_tmpdir' not given, the dataset will contain different samples at every epoch due to 'random_frames', this will break assumptions made by Keras")
            tmp_fn = make_random_frame_chunker_fn(frame_config["length"])
            frame_chunker_fn = lambda feats, *args: (tmp_fn(feats), *args)
            ds = ds.map(frame_chunker_fn, num_parallel_calls=TF_AUTOTUNE)
        if verbosity:
            print("Flattening all utterance chunks into independent utterances")
        def unbatch_ragged_frames(frames, meta):
            frames_ds = tf.data.Dataset.from_tensor_slices(frames)
            # Create new uttids by appending enumeration of chunks onto original uttid, separated by '-'
            def append_enumeration(i, str_meta):
                chunk_num_str = tf.strings.as_string(i, width=6, fill='0')
                new_uttid = tf.strings.join((str_meta[0], chunk_num_str), separator='-')
                return tf.concat((tf.expand_dims(new_uttid, 0), str_meta[1:]), axis=0)
            str_meta_ds = (tf.data.Dataset
                          .from_tensors(meta[0])
                          .repeat()
                          .enumerate(start=1)
                          .map(append_enumeration))
            # Repeat all other metadata indefinitely
            rest_ds = [tf.data.Dataset.from_tensors(m).repeat() for m in meta[1:]]
            return tf.data.Dataset.zip((frames_ds, str_meta_ds, *rest_ds))
        def is_not_empty(features, *args):
            return tf.shape(features)[0] > 0
        # TODO get correct signal chunks after partitioning feature frames (nontrivial)
        def drop_wavs(features, str_meta, *args):
            return features, str_meta, audio_feat.Wav(tf.zeros([1]), 16000)
        # TODO replacing flat_map with tf.data.Dataset.interleave would allow parallelism,
        # but then the order of utterances will not be deterministic ('stability' could be a config flag)
        if verbosity:
            print("Warning: dropping original signals since they cannot be matched to the feature frames")
        ds = (ds.flat_map(unbatch_ragged_frames)
                .filter(is_not_empty)
                .map(drop_wavs))
    else:
        ds = ds.map(lambda feats, meta: (feats, *meta))
    # Transform dataset such that 2 first elements will always be (sample, onehot_label) and rest will be metadata that can be safely dropped when training starts
    def to_model_input(feats, str_meta, *rest):
        uttid, label = str_meta[0], str_meta[1]
        return (feats, label2onehot(label), uttid, *rest)
    ds = ds.map(to_model_input, num_parallel_calls=TF_AUTOTUNE)
    if "min_shape" in config:
        if verbosity:
            print("Filtering features by minimum shape", config["min_shape"])
        ds = filter_with_min_shape(ds, config["min_shape"])
    shuffle_buffer_size = config.get("shuffle_buffer", {"before_cache": 0})["before_cache"]
    if shuffle_buffer_size:
        if verbosity:
            print("Shuffling features with shuffle buffer size", shuffle_buffer_size)
        ds = ds.shuffle(shuffle_buffer_size)
    if "padded_batch" in config:
        pad_kwargs = config["padded_batch"]["kwargs"]
        if verbosity:
            print("Batching features with padded batch kwargs:")
            yaml_pprint(pad_kwargs)
        pad_kwargs["padded_shapes"] = tuple(pad_kwargs["padded_shapes"])
        pad_kwargs["padding_values"] = tuple(tf.constant(float(val), dtype=tf.float32) for val in pad_kwargs["padding_values"])
        ds = without_metadata(ds).padded_batch(**pad_kwargs)
    elif "batch_size" in config:
        if verbosity:
            print("Batching features with batch size", config["batch_size"])
        ds = ds.batch(config["batch_size"], drop_remainder=True)
    if "bucket_by_sequence_length" in config:
        if verbosity:
            print("Batching features by bucketing samples into fixed length, padded sequence length buckets")
        seq_len_fn = lambda feats, meta: tf.shape(feats)[0]
        bucket_conf = config["bucket_by_sequence_length"]
        bucket_boundaries = np.linspace(
            bucket_conf["bins"]["min"],
            bucket_conf["bins"]["max"],
            bucket_conf["bins"]["num"],
            dtype=np.int32)
        bucket_batch_sizes = [1] + (len(bucket_boundaries) - 1) * [bucket_conf["batch_size"]] + [1]
        bucketing_fn = tf.data.experimental.bucket_by_sequence_length(
            seq_len_fn,
            bucket_boundaries,
            bucket_batch_sizes,
            **bucket_conf.get("kwargs", {}))
        ds = ds.apply(bucketing_fn)
    elif "group_by_sequence_length" in config:
        ds = group_by_sequence_length(
                ds,
                config["group_by_sequence_length"].get("min_batch_size", 0),
                config["group_by_sequence_length"]["max_batch_size"],
                verbosity=verbosity,
                sequence_dim=0)
    if config.get("cache_prepared_dataset_to_tmpdir", False):
        tmp_cache_path = "{}/tensorflow-cache/{}/training-prepared_{}_{}".format(
                os.environ.get("TMPDIR", "/tmp"),
                model_id,
                int(time.time()),
                conf_checksum)
        if verbosity:
            print("Caching prepared dataset iterator to '{}'".format(tmp_cache_path))
        os.makedirs(os.path.dirname(tmp_cache_path), exist_ok=True)
        ds = ds.cache(filename=tmp_cache_path)
        cache_shuffle_buffer_size = config.get("shuffle_buffer", {"after_cache": 0})["after_cache"]
        if cache_shuffle_buffer_size:
            if verbosity:
                print("Shuffling cached features with shuffle buffer size", cache_shuffle_buffer_size)
            ds = ds.shuffle(cache_shuffle_buffer_size)
    # assume autotuned prefetch (turned off when config["prefetch"] is None)
    if "prefetch" not in config:
        if verbosity:
            print("Using autotune value", TF_AUTOTUNE, "for prefetching batches")
        ds = ds.prefetch(TF_AUTOTUNE)
    elif config["prefetch"] is not None:
        if verbosity:
            print("Using fixed size prefetch value", config["prefetch"])
        ds = ds.prefetch(config["prefetch"])
    return ds

#TODO histogram support (would use significantly less space than images+audio)
def attach_dataset_logger(ds, features_name, max_outputs=10, image_resize_kwargs=None, colormap="viridis", num_batches=-1):
    """
    Write Tensorboard summary information for samples in the given tf.data.Dataset.
    """
    # TF colormap trickery from https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
    # The idea is to extract all RGB values from the matplotlib colormap into a tf.constant
    cmap = matplotlib.cm.get_cmap(colormap)
    num_colors = cmap.N
    colors = tf.constant(cmap(np.arange(num_colors + 1))[:,:3], dtype=tf.float32)
    if image_resize_kwargs is None:
        image_resize_kwargs = {"size_multiplier": 0}
    img_size_multiplier = tf.constant(image_resize_kwargs.pop("size_multiplier", 1), dtype=tf.float32)
    @tf.function
    def inspect_batches(batch_idx, batch):
        samples, labels, uttids, wavs = batch[:4]
        # Scale features between 0 and 1 to produce a grayscale image
        image = feature_scaling(samples, tf.constant(0.0), tf.constant(1.0))
        # Map linear colormap over all grayscale values [0, 1] to produce an RGB image
        indices = tf.dtypes.cast(tf.math.round(image * num_colors), tf.int32)
        image = tf.gather(colors, indices)
        # Prepare for tensorboard
        image = tf.image.transpose(image)
        image = tf.image.flip_up_down(image)
        if img_size_multiplier > 0:
            old_size = tf.cast(tf.shape(image)[1:3], tf.float32)
            new_size = tf.cast(img_size_multiplier * old_size, tf.int32)
            image = tf.image.resize(image, new_size, **image_resize_kwargs)
        tf.debugging.assert_all_finite(image, message="non-finite values in image when trying to create dataset logger for tensorboard")
        tf.summary.histogram("input_samples", samples, step=batch_idx)
        tf.summary.histogram("input_labels", labels, step=batch_idx)
        tf.summary.image(features_name, image, step=batch_idx, max_outputs=max_outputs)
        tf.debugging.assert_equal(tf.expand_dims(wavs.sample_rate[0], 0), wavs.sample_rate, message="All utterances in a batch must have the same sample rate")
        if tf.reduce_all(tf.size(wavs.audio) > 0) and tf.reduce_all(wavs.sample_rate != 0):
            tf.summary.audio("utterances", tf.expand_dims(wavs.audio, -1), wavs.sample_rate[0], step=batch_idx, max_outputs=max_outputs)
        enumerated_uttids = tf.strings.reduce_join(
                (tf.strings.as_string(tf.range(1, max_outputs + 1)), uttids[:max_outputs]),
                axis=0,
                separator=": ")
        tf.summary.text("utterance_ids", enumerated_uttids, step=batch_idx)
        return batch
    return (ds.take(num_batches)
              .enumerate()
              .map(inspect_batches))

def without_metadata(dataset):
    return dataset.map(lambda feats, inputs, *meta: (feats, inputs))

def get_chunk_loader(wav_config, verbosity, datagroup_key):
    chunk_config = wav_config["chunks"]
    target_sr = wav_config["filter_sample_rate"]
    # Deep copy all augmentation config dicts because we will be mutating them soon
    augment_config = json.loads(json.dumps(wav_config.get("augmentation", [])))
    if datagroup_key != "train":
        if verbosity:
            print("skipping augmentation due to non-training datagroup: '{}'".format(datagroup_key), file=sys.stderr)
        augment_config = []
    for conf in augment_config:
        # prepare noise augmentation
        if conf["type"] == "additive_noise":
            noise_source_dir = conf["noise_source"]
            conf["noise_source"] = {}
            with open(os.path.join(noise_source_dir, "id2label")) as f:
                id2label = dict(l.strip().split() for l in f)
            label2path = collections.defaultdict(list)
            with open(os.path.join(noise_source_dir, "id2path")) as f:
                for noise_id, path in (l.strip().split() for l in f):
                    label2path[id2label[noise_id]].append((noise_id, path))
            tmpdir = conf.get("copy_to_tmpdir")
            if tmpdir:
                if os.path.isdir(tmpdir):
                    if verbosity:
                        print("tmpdir for noise source files given, but it already exists at '{}', so copying will be skipped".format(tmpdir))
                else:
                    if verbosity:
                        print("copying all noise source wavs to '{}'".format(tmpdir))
                    os.makedirs(tmpdir)
                    tmp = collections.defaultdict(list)
                    for noise_type, noise_paths in label2path.items():
                        for noise_id, path in noise_paths:
                            new_path = os.path.join(tmpdir, noise_id + ".wav")
                            shutil.copy2(path, new_path)
                            if verbosity > 3:
                                print(" ", path, "->", new_path)
                            tmp[noise_type].append((noise_id, new_path))
                    label2path = tmp
            for noise_type, noise_paths in label2path.items():
                conf["noise_source"][noise_type] = [path for _, path in noise_paths]
    chunk_len_seconds = 1e-3 * chunk_config["length_ms"]
    chunk_step_seconds = 1e-3 * chunk_config["step_ms"]
    max_pad_seconds = 1e-3 * chunk_config.get("max_pad_ms", 0)
    def chunker(signal, sr, meta):
        chunk_len = int(sr * chunk_len_seconds)
        max_pad = int(sr * max_pad_seconds)
        if signal.size + max_pad < chunk_len:
            return
        chunk_step = int(sr * chunk_step_seconds)
        num_full_chunks = max(0, 1 + (signal.size - chunk_len) // chunk_step)
        last_chunk_len = signal.size - num_full_chunks * chunk_step
        if last_chunk_len < chunk_len <= last_chunk_len + max_pad:
            signal = np.pad(signal, [0, chunk_len - last_chunk_len])
        chunks = librosa.util.frame(signal, chunk_len, chunk_step, axis=0)
        uttid, label = meta[:2]
        for i, chunk in enumerate(chunks, start=1):
            chunk_id = "{:s}-{:06d}".format(uttid, i)
            yield (chunk, sr), chunk_id, label
    def chunk_loader(signal, sr, meta, *args):
        chunk_length = int(sr * chunk_len_seconds)
        max_pad = int(sr * max_pad_seconds)
        if signal.size + max_pad < chunk_length:
            if verbosity:
                tf_util.tf_print("skipping too short signal (min chunk length is ", chunk_length, " + ", max_pad, " max_pad): length ", signal.size, ", utt ", meta[0], output_stream=sys.stderr, sep='')
            return
        uttid = meta[0].decode("utf-8")
        yield from chunker(signal, target_sr, (uttid, *meta[1:]))
        rng = np.random.default_rng()
        for conf in augment_config:
            if conf["type"] == "random_resampling":
                # apply naive speed modification by resampling
                ratio = rng.uniform(conf["range"][0], conf["range"][1])
                with contextlib.closing(io.BytesIO()) as buf:
                    soundfile.write(buf, signal, int(ratio * target_sr), format="WAV")
                    buf.seek(0)
                    resampled_signal, _ = librosa.core.load(buf, sr=target_sr, mono=True)
                new_uttid = "{:s}-speed{:.3f}".format(uttid, ratio)
                new_meta = (new_uttid, *meta[1:])
                yield from chunker(resampled_signal, target_sr, new_meta)
            elif conf["type"] == "additive_noise":
                for noise_type, db_min, db_max in conf["snr_def"]:
                    noise_signal = np.zeros(0, dtype=signal.dtype)
                    noise_paths = []
                    while noise_signal.size < signal.size:
                        rand_noise_path = rng.choice(conf["noise_source"][noise_type])
                        noise_paths.append(rand_noise_path)
                        sig, _ = librosa.core.load(rand_noise_path, sr=target_sr, mono=True)
                        noise_signal = np.concatenate((noise_signal, sig))
                    noise_begin = rng.integers(0, noise_signal.size - signal.size, endpoint=True)
                    noise_signal = noise_signal[noise_begin:noise_begin+signal.size]
                    snr_db = rng.integers(db_min, db_max, endpoint=True)
                    clean_and_noise = audio_feat.numpy_snr_mixer(signal, noise_signal, snr_db)[2]
                    new_uttid = "{:s}-{:s}_snr{:d}".format(uttid, noise_type, snr_db)
                    if not np.all(np.isfinite(clean_and_noise)):
                        if verbosity:
                            tf_util.tf_print("warning: snr_mixer failed, augmented signal ", new_uttid, " has non-finite values and will be skipped. Utterance was ", uttid, ", and chosen noise signals were\n  ", tf.strings.join(noise_paths, separator="\n  "), output_stream=sys.stderr, sep='')
                        return
                    new_meta = (new_uttid, *meta[1:])
                    yield from chunker(clean_and_noise, target_sr, new_meta)
    if verbosity:
        tf_util.tf_print("Using wav chunk loader, generating chunks of length {} with step size {} (milliseconds)".format(chunk_config["length_ms"], chunk_config["step_ms"]))
    return chunk_loader

def filter_wavs_with_webrtcvad(dataset, config, min_duration_sec, verbosity):
    assert config["frame_ms"] in (10, 20, 30), "webrtcvad supports only vad frame lengths of 10, 20, or 30 ms"
    assert 0 <= config["aggressiveness"] <= 3, "webrtcvad aggressiveness values must be in range [0, 3]"
    aggressiveness = config["aggressiveness"]
    frame_sec = tf.constant(1e-3 * config["frame_ms"], tf.float32)
    min_non_speech_frames = config["min_non_speech_length_ms"] // config["frame_ms"]
    def get_vad_decisions(wav, vad_frame_length):
        _, pcm_data = audio_feat.wav_to_pcm_data(wav.audio, wav.sample_rate)
        args = (wav.audio,
                wav.sample_rate,
                pcm_data,
                vad_frame_length,
                aggressiveness,
                min_non_speech_frames)
        return tf.numpy_function(audio_feat.numpy_fn_get_webrtcvad_decisions, args, tf.bool)
    def drop_silent_frames(wav, *rest):
        vad_frame_length = tf.cast(tf.cast(wav.sample_rate, tf.float32) * frame_sec, tf.int32)
        frames = tf.signal.frame(wav.audio, vad_frame_length, vad_frame_length, axis=0)
        vad_decisions = get_vad_decisions(wav, vad_frame_length)
        vad_decisions = tf.reshape(vad_decisions, [tf.shape(frames)[0]])
        voiced_signal = tf.reshape(frames[vad_decisions], [-1])
        voiced_wav = audio_feat.Wav(voiced_signal, wav.sample_rate)
        return (voiced_wav, *rest)
    def has_min_chunk_length(wav, meta):
        min_wav_length = tf.cast(tf.cast(wav.sample_rate, tf.float32) * min_duration_sec, tf.int32)
        wav_length = tf.size(wav.audio)
        ok = wav_length >= min_wav_length
        if verbosity > 1 and not ok:
            tf_util.tf_print("dropping utterance ", meta[0], ", reason: after VAD, signal samples ", wav_length, " < ", min_wav_length, " min chunk samples", sep='', output_stream=sys.stderr)
        return ok
    return (dataset
            .map(drop_silent_frames, num_parallel_calls=TF_AUTOTUNE)
            .filter(has_min_chunk_length))

def get_random_chunk_loader(paths, meta, wav_config, verbosity=0):
    raise NotImplementedError("todo, rewrite to return a function that supports tf.data.Dataset.interleave, like in get_chunk_loader")
    chunk_config = wav_config["wav_to_random_chunks"]
    sample_rate = wav_config["filter_sample_rate"]
    lengths = tf.cast(
        tf.linspace(
            float(sample_rate * chunk_config["length"]["min"]),
            float(sample_rate * chunk_config["length"]["max"]),
            int(chunk_config["length"]["num_bins"]),
        ),
        tf.int32
    )
    def get_random_length():
        rand_index = tf.random.uniform([], 0, tf.size(lengths), dtype=tf.int32)
        rand_len = tf.gather(lengths, rand_index)
        return rand_len
    overlap_ratio = float(chunk_config["length"]["overlap_ratio"])
    assert overlap_ratio < 1.0
    min_chunk_length = int(sample_rate * chunk_config["min_chunk_length"])
    assert min_chunk_length > 0, "invalid min chunk length"
    def random_chunk_loader():
        for p, *m in zip(paths, meta):
            wav = audio_feat.read_wav(tf.constant(p, tf.string))
            if "filter_sample_rate" in wav_config and wav.sample_rate != wav_config["filter_sample_rate"]:
                if verbosity:
                    print("skipping file", p, ", it has a sample rate", wav.sample_rate, "but config has 'filter_sample_rate'", wav_config["filter_sample_rate"], file=sys.stderr)
                continue
            begin = 0
            rand_len = get_random_length()
            chunk = wav.audio[begin:begin+rand_len]
            while len(chunk) >= min_chunk_length:
                yield (audio_feat.Wav(chunk, wav.sample_rate), *m)
                begin += round((1.0 - overlap_ratio) * float(rand_len))
                rand_len = get_random_length()
                chunk = wav.audio[begin:begin+rand_len]
    if verbosity:
        tf_util.tf_print("Using random wav chunk loader, drawing lengths (in frames) from", lengths, "with", overlap_ratio, "overlap ratio and", min_chunk_length, "minimum chunk length")
    return random_chunk_loader

# Use batch_size > 1 iff _every_ audio file in paths has the same amount of samples
# TODO: fix this mess
def extract_features_from_paths(feat_config, paths, meta, datagroup_key, verbosity=0):
    paths, meta, durations = list(paths), [m[:3] for m in meta], [m[3] for m in meta]
    assert len(paths) == len(meta) == len(durations), "Cannot extract features from paths when the amount of metadata {} and durations {} does not match the amount of wavfile paths {}".format(len(meta), len(durations), len(paths))
    wav_config = feat_config.get("wav_config", {})
    paths_t = tf.constant(paths, tf.string)
    meta_t = tf.constant(meta, tf.string)
    duration_t = tf.constant(durations, tf.float32)
    target_sr = wav_config.get("filter_sample_rate", -1)
    if verbosity:
        if target_sr < 0:
            print("filter_sample_rate not defined in wav_config, wav files will not be filtered")
        else:
            print("filter_sample_rate set to {}, wav files with mismatching sample rates will be ignored and no features will be extracted from those files".format(target_sr))
    def read_wav_with_meta(path, meta, duration):
        return audio_feat.read_wav(path), meta
    def is_not_empty(wav, meta):
        ok = tf.size(wav.audio) > 0
        if verbosity and not ok:
            tf_util.tf_print("dropping utterance ", meta[0], ", reason: empty signal", sep='', output_stream=sys.stderr)
        return ok
    def has_target_sample_rate(wav, meta):
        ok = target_sr == -1 or wav.sample_rate == target_sr
        if verbosity and not ok:
            tf_util.tf_print("dropping utterance ", meta[0], ", reason: sample rate ", wav.sample_rate, " != ", target_sr, " target sample rate", sep='', output_stream=sys.stderr)
        return ok
    # Dataset that reads all wav files from the given paths
    wavs_ds = (tf.data.Dataset
            .from_tensor_slices((paths_t, meta_t, duration_t))
            .map(read_wav_with_meta, num_parallel_calls=TF_AUTOTUNE)
            .filter(is_not_empty)
            .filter(has_target_sample_rate))
    webrtcvad_config = wav_config.get("webrtcvad")
    if webrtcvad_config:
        if verbosity:
            print("WebRTC VAD config defined in wav_config, all valid signals will be filtered to remove silent segments of at least {} ms".format(webrtcvad_config["min_non_speech_length_ms"]))
        min_duration = 0
        wavs_ds = filter_wavs_with_webrtcvad(wavs_ds, webrtcvad_config, min_duration, verbosity)
    if "chunks" in wav_config:
        dataset_types = (
            (tf.float32, tf.int32),
            tf.string,
            tf.string)
        dataset_shapes = (
            (tf.TensorShape([None]), tf.TensorShape([])),
            tf.TensorShape([]),
            tf.TensorShape([]))
        expected_samples_per_file = wav_config.get("expected_samples_per_file", 1)
        if verbosity:
            print("Loading samples from wavs as fixed sized chunks, expecting up to {} consecutive elements from each file".format(expected_samples_per_file))
            if verbosity > 1:
                print("Expecting dataset types:", dataset_types)
                print("Expecting dataset shapes:", dataset_shapes)
        chunk_loader_fn = get_chunk_loader(wav_config, verbosity, datagroup_key)
        def chunk_dataset_generator(*args):
            return tf.data.Dataset.from_generator(
                chunk_loader_fn,
                dataset_types,
                dataset_shapes,
                args=args)
        def unpack_wav_tuples(wav, meta):
            return wav.audio, wav.sample_rate, meta
        def pack_wav_tuples(signal, sample_rate, meta):
            return audio_feat.Wav(signal, sample_rate), meta
        if verbosity:
            print("Generating chunks from all source audio files")
        wavs_ds = (wavs_ds
                    .map(unpack_wav_tuples, num_parallel_calls=TF_AUTOTUNE)
                    .interleave(
                        chunk_dataset_generator,
                        block_length=expected_samples_per_file,
                        num_parallel_calls=TF_AUTOTUNE)
                    .map(pack_wav_tuples, num_parallel_calls=TF_AUTOTUNE))
    if "batch_wavs_by_length" in feat_config:
        window_size = feat_config["batch_wavs_by_length"]["max_batch_size"]
        if verbosity:
            print("Batching all wavs by equal length into batches of max size {}".format(window_size))
        key_fn = lambda wav, *meta: tf.cast(tf.size(wav.audio), tf.int64)
        reduce_fn = lambda key, group_ds: group_ds.batch(window_size)
        group_by_wav_length = tf.data.experimental.group_by_window(key_fn, reduce_fn, window_size)
        wavs_batched = wavs_ds.apply(group_by_wav_length)
    else:
        batch_size = feat_config.get("batch_size", 1)
        if verbosity:
            print("Batching wavs with batch size", batch_size)
        wavs_batched = wavs_ds.batch(batch_size)
    if verbosity:
        print("Applying feature extractor to batched wavs")
    feat_extract_args = feat_extraction_args_as_list(feat_config)
    def wav_batch_to_features(wavs, *meta):
        return extract_features(wavs, *feat_extract_args), (*meta, wavs)
    return wavs_batched.map(wav_batch_to_features, num_parallel_calls=TF_AUTOTUNE)

def parse_kaldi_features(utterance_list, features_path, utt2meta, expected_shape, feat_conf):
    utt2label = {u: d["label"] for u, d in utt2meta.items()}
    utt2feats = kaldiio.load_scp(features_path)
    feat_conf = dict(feat_conf)
    normalize_mean_axis = feat_conf.pop("normalize_mean_axis", None)
    normalize_stddev_axis = feat_conf.pop("normalize_stddev_axis", None)
    def assert_shape(shape):
        shape_str = "{} vs {}".format(shape, expected_shape)
        assert len(shape) == len(expected_shape), shape_str
        assert all(x == y for x, y in zip(shape, expected_shape) if y is not None), shape_str
    def datagen():
        for utt in utterance_list:
            if utt not in utt2feats:
                print("warning: skipping utterance '{}' since it is not in the kaldi scp file".format(utt), file=sys.stderr)
                continue
            feats = utt2feats[utt]
            #assert_shape(feats.shape)
            normalized = tf.constant(feats, tf.float32)
            if normalize_mean_axis is not None:
                mean = tf.math.reduce_mean(feats, axis=normalize_mean_axis, keepdims=True)
                normalized = feats - mean
            if normalize_stddev_axis is not None:
                stddev = tf.math.reduce_std(feats, axis=normalize_stddev_axis, keepdims=True)
                normalized = tf.math.divide_no_nan(normalized, stddev)
            yield normalized, (utt, utt2label[utt])
    ds = tf.data.Dataset.from_generator(
        datagen,
        (tf.float32, tf.string),
        (tf.TensorShape(expected_shape), tf.TensorShape([2])))
    # Add empty signals (since we don't know the origin of these features)
    empty_wav = audio_feat.Wav(tf.zeros([0]), 0)
    add_empty_signal = lambda feats, meta: (feats, (meta, empty_wav))
    return ds.map(add_empty_signal).batch(1)
