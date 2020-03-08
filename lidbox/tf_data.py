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

from . import yaml_pprint
from . import audio_feat
from .tf_util import tf_print

debug = False
if debug:
    TF_AUTOTUNE = None
    tf.autograph.set_verbosity(10, alsologtostdout=True)
else:
    TF_AUTOTUNE = tf.data.experimental.AUTOTUNE


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

@tf.function
def load_wav(path):
    wav = tf.audio.decode_wav(tf.io.read_file(path))
    # Merge channels by averaging, for mono this just drops the channel dim.
    return audio_feat.Wav(tf.math.reduce_mean(wav.audio, axis=1, keepdims=False), wav.sample_rate)

# Copied from
# https://github.com/wiseman/py-webrtcvad/blob/fe9d953217932c319070a6aeeeb6860aeb1c474e/example.py
def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

@tf.function
def write_wav(path, wav):
    tf.debugging.assert_rank(wav, 2, "write_wav expects signals with shape [N, c] where N is amount of samples and c channels.")
    return tf.io.write_file(path, tf.audio.encode_wav(wav.audio, wav.sample_rate))

def count_dataset(ds):
    return ds.reduce(tf.constant(0, tf.int64), lambda c, elem: c + 1)

def filter_with_min_shape(ds, min_shape, ds_index=0):
    min_shape = tf.constant(min_shape, dtype=tf.int32)
    at_least_min_shape = lambda *t: tf.math.reduce_all(tf.shape(t[ds_index]) >= min_shape)
    return ds.filter(at_least_min_shape)

def append_webrtcvad_decisions(ds, config):
    assert 0 <= config["aggressiveness"] <= 3, "webrtcvad aggressiveness values must be in range [0, 3]"
    # TODO assert encoded frame lengths are in {10, 20, 30} ms since webrtcvad supports only those
    aggressiveness = tf.constant(config["aggressiveness"], tf.int32)
    vad_frame_length = tf.constant(config["vad_frame_length"], tf.int32)
    vad_frame_step = tf.constant(config["vad_frame_step"], tf.int32)
    feat_frame_length = tf.constant(config["feat_frame_length"], tf.int32)
    feat_frame_step = tf.constant(config["feat_frame_step"], tf.int32)
    wavs_to_bytes = lambda wav, *rest: (wav, audio_feat.wav_to_bytes(wav), *rest)
    apply_webrtcvad = lambda wav, wav_bytes, *rest: (
        wav,
        *rest,
        tf.numpy_function(
            audio_feat.framewise_webrtcvad_decisions,
            [tf.size(wav.audio),
             wav_bytes,
             wav.sample_rate,
             vad_frame_length,
             vad_frame_step,
             feat_frame_length,
             feat_frame_step,
             aggressiveness],
            tf.bool))
    return (ds.map(wavs_to_bytes, num_parallel_calls=TF_AUTOTUNE)
              .map(apply_webrtcvad, num_parallel_calls=TF_AUTOTUNE))

def append_mfcc_energy_vad_decisions(ds, config):
    vad_kwargs = dict(config)
    spec_kwargs = vad_kwargs.pop("spectrogram")
    melspec_kwargs = vad_kwargs.pop("melspectrogram")
    f = lambda wav, *rest: (
        tf_print("mfcc vad", rest[0]) and wav,
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
        chunk_indices = tf.ragged.range(begin, end)
        return tf.gather(features, chunk_indices)
    return chunk_timedim_randomly

def prepare_dataset_for_training(ds, config, feat_config, label2onehot, model_id, conf_checksum='', verbosity=0):
    if "frames" in config:
        raise NotImplementedError("todo")
        if verbosity:
            print("Dividing features time dimension into frames")
        assert "convert_to_images" not in config, "todo, time dim random chunks for image data"
        # frame_axis = 1 if "convert_to_images" in config else 0
        # if frame_axis == 1:
            # ds = ds.map(lambda f, *meta: (tf.transpose(f, perm=(1, 0, 2, 3)), *meta))
        if config["frames"].get("random", False):
            if verbosity:
                print("Dividing features time dimension randomly")
            assert isinstance(config["frames"]["length"], dict), "key 'frames.length' must map to a dict type when doing random chunking of frames"
            frame_chunker_fn = make_random_frame_chunker_fn(config["frames"]["length"])
            chunk_timedim_randomly = lambda f, *meta: (frame_chunker_fn(f), *meta)
            ds = ds.map(chunk_timedim_randomly, num_parallel_calls=TF_AUTOTUNE)
        else:
            if verbosity:
                print("Dividing features time dimension into fixed length chunks")
            # Extract frames from all features, using the same metadata for each frame of one sample of features
            seq_len = config["frames"]["length"]
            seq_step = config["frames"]["step"]
            pad_zeros = config["frames"].get("pad_zeros", False)
            to_frames = lambda feats, *meta: (
                tf.signal.frame(feats, seq_len, seq_step, pad_end=pad_zeros, axis=0),
                *meta)
            ds = ds.map(to_frames)
        if config["frames"].get("flatten", True):
            def _unbatch_ragged_frames(frames, *meta):
                frames_ds = tf.data.Dataset.from_tensor_slices(frames)
                inf_repeated_meta_ds = [tf.data.Dataset.from_tensors(m).repeat() for m in meta]
                return tf.data.Dataset.zip((frames_ds, *inf_repeated_meta_ds))
            ds = ds.flat_map(_unbatch_ragged_frames)
        ds = ds.filter(lambda frames, *meta: tf.shape(frames)[0] > 0)
        if "normalize" in config["frames"]:
            axis = config["frames"]["normalize"]["axis"]
            if verbosity:
                print("Normalizing means frame-wise over axis {}".format(axis))
            def normalize_frames(frames, *meta):
                return (frames - tf.math.reduce_mean(frames, axis=axis, keepdims=True), *meta)
            ds = ds.map(normalize_frames)
    # Transform dataset such that 2 first elements will always be (sample, onehot_label) and rest will be metadata that can be safely dropped when training starts
    to_model_input = lambda feats, meta: (feats, label2onehot(meta[1]), meta[0], *meta[2:])
    ds = ds.map(to_model_input)
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
        max_batch_size = tf.constant(config["group_by_sequence_length"]["max_batch_size"], tf.int64)
        if verbosity:
            tf_print("Grouping samples by sequence length into batches of max size", max_batch_size)
        get_seq_len = lambda feat, meta: tf.cast(tf.shape(feat)[0], tf.int64)
        group_to_batch = lambda key, group: group.batch(max_batch_size)
        ds = ds.apply(tf.data.experimental.group_by_window(get_seq_len, group_to_batch, window_size=max_batch_size))
        if "min_batch_size" in config["group_by_sequence_length"]:
            min_batch_size = config["group_by_sequence_length"]["min_batch_size"]
            if verbosity:
                print("Dropping batches smaller than min_batch_size", min_batch_size)
            min_batch_size = tf.constant(min_batch_size, tf.int32)
            ds = ds.filter(lambda batch, meta: (tf.shape(batch)[0] >= min_batch_size))
    if config.get("copy_cache_to_tmp", False):
        tmp_cache_path = "/tmp/tensorflow-cache/{}/training-prepared_{}_{}".format(model_id, int(time.time()), conf_checksum)
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

#TODO histogram support (uses significantly less space than images+audio)
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

# Copied from
# https://github.com/microsoft/MS-SNSD/blob/e84aba38cac499a109c0d237a00dc600dcf9b7e7/audiolib.py
def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = np.sqrt((clean**2).mean())
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = np.sqrt((clean**2).mean())
    rmsnoise = np.sqrt((noise**2).mean())
    scalarnoise = 10 ** (-25 / 20) / rmsnoise
    noise = noise * scalarnoise
    rmsnoise = np.sqrt((noise**2).mean())
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech

def get_chunk_loader(wav_config, verbosity, datagroup_key):
    chunks = wav_config["chunks"]
    target_sr = wav_config.get("filter_sample_rate")
    # Deep copy all augmentation config dicts because we will be mutating them soon
    augment_config = json.loads(json.dumps(wav_config.get("augmentation", [])))
    if datagroup_key != "train":
        if verbosity:
            print("skipping augmentation due to non-training datagroup: '{}'".format(datagroup_key), file=sys.stderr)
        augment_config = []
    vad_config = wav_config.get("webrtcvad")
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

    def drop_silence(signal, sr):
        vad_frame_ms = vad_config["frame_ms"]
        assert vad_frame_ms in (10, 20, 30)
        assert sr == target_sr, "unexpected sample rate {}, cannot do vad_config because wav write would distort pitch".format(sr)
        with contextlib.closing(io.BytesIO()) as buf:
            soundfile.write(buf, signal, sr, format="WAV")
            buf.seek(0)
            pcm_data, sr = read_wave(buf)
        assert sr == target_sr, "vad_config failed during WAV read"
        step = int(sr * 1e-3 * vad_frame_ms * 2)
        if signal.size < step//2:
            return np.zeros(0, dtype=signal.dtype)
        frames = librosa.util.frame(signal, step//2, step//2, axis=0)
        vad_decisions = np.ones(frames.shape[0], dtype=np.bool)
        min_non_speech_frames = vad_config["min_non_speech_length_ms"] // vad_frame_ms
        vad = webrtcvad.Vad(vad_config["aggressiveness"])
        non_speech_begin = -1
        for f, i in enumerate(range(0, len(pcm_data) - len(pcm_data) % step, step)):
            if not vad.is_speech(pcm_data[i:i+step], sr):
                vad_decisions[f] = False
                if non_speech_begin < 0:
                    non_speech_begin = f
            else:
                if non_speech_begin >= 0 and f - non_speech_begin < min_non_speech_frames:
                    # too short non-speech segment, revert all non-speech decisions up to f
                    vad_decisions[np.arange(non_speech_begin, f)] = True
                non_speech_begin = -1
        voiced_frames = frames[vad_decisions]
        if voiced_frames.size > 0:
            voiced_signal = np.concatenate(voiced_frames, axis=0)
        else:
            voiced_signal = np.zeros(0, dtype=signal.dtype)
        if verbosity > 3:
            print("dropping {} frames due to vad, signal shape {} voiced_signal shape {}".format(int((~vad_decisions).sum()), signal.shape, voiced_signal.shape))
        return voiced_signal
    def chunker(signal, sr, meta):
        chunk_len = int(sr * 1e-3 * chunks["length_ms"])
        chunk_step = int(sr * 1e-3 * chunks["step_ms"])
        if signal.size >= chunk_len:
            for i, chunk in enumerate(librosa.util.frame(signal, chunk_len, chunk_step, axis=0)):
                chunk_uttid = tf.strings.join((meta[0], "-{:06d}".format(i)))
                yield (chunk, sr), chunk_uttid, meta[1]
    def chunk_loader(original_signal, sr, wav_path, meta):
        chunk_stats = []
        utt, label, dataset = meta[:3]
        # original_signal, sr = librosa.core.load(wav_path, sr=target_sr, mono=True)
        if vad_config:
            original_signal = drop_silence(original_signal, sr)
        chunk_length = int(sr * 1e-3 * chunks["length_ms"])
        if original_signal.size < chunk_length:
            if verbosity:
                tf_print("skipping too short signal (min chunk length is ", chunk_length, "): length ", original_signal.size, ", path ", tf.constant(wav_path, tf.string), output_stream=sys.stderr, sep='')
            return
        num_chunks_produced = 0
        time_begin = time.perf_counter()
        for c in chunker(original_signal, target_sr, meta):
            num_chunks_produced += 1
            yield c
        time_end = time.perf_counter()
        chunk_stats.append(("original-signal", num_chunks_produced, time_end - time_begin))
        for conf in augment_config:
            if "datasets_include" in conf and dataset not in conf["datasets_include"]:
                continue
            if "datasets_exclude" in conf and dataset in conf["datasets_exclude"]:
                continue
            #TODO preload whole augmentation dataset into memory
            if conf["type"] == "random_resampling":
                # apply naive speed modification by resampling
                rate = np.random.uniform(conf["range"][0], conf["range"][1])
                with contextlib.closing(io.BytesIO()) as buf:
                    soundfile.write(buf, original_signal, int(rate * target_sr), format="WAV")
                    buf.seek(0)
                    signal, _ = librosa.core.load(buf, sr=target_sr, mono=True)
                new_uttid = tf.strings.join((utt, "-speed{:.3f}".format(rate)))
                num_chunks_produced = 0
                time_begin = time.perf_counter()
                for c in chunker(signal, target_sr, (new_uttid, *meta[1:])):
                    num_chunks_produced += 1
                    yield c
                time_end = time.perf_counter()
                chunk_stats.append(("resampling", num_chunks_produced, time_end - time_begin))
            elif conf["type"] == "additive_noise":
                for noise_type, db_min, db_max in conf["snr_def"]:
                    noise_signal = np.zeros(0, dtype=original_signal.dtype)
                    noise_paths = []
                    while noise_signal.size < original_signal.size:
                        rand_noise_path = random.choice(conf["noise_source"][noise_type])
                        noise_paths.append(rand_noise_path)
                        sig, _ = librosa.core.load(rand_noise_path, sr=target_sr, mono=True)
                        noise_signal = np.concatenate((noise_signal, sig))
                    noise_begin = random.randint(0, noise_signal.size - original_signal.size)
                    noise_signal = noise_signal[noise_begin:noise_begin+original_signal.size]
                    snr_db = random.randint(db_min, db_max)
                    clean, noise, clean_and_noise = snr_mixer(original_signal, noise_signal, snr_db)
                    new_uttid = tf.strings.join((utt, "-{:s}_snr{:d}".format(noise_type, snr_db)))
                    if not np.all(np.isfinite(clean_and_noise)):
                        if verbosity:
                            tf_print("warning: snr_mixer failed, augmented signal ", new_uttid, " has non-finite values and will be skipped. Utterance source was ", wav_path, ", and chosen noise signals were\n  ", tf.strings.join(noise_paths, separator="\n  "), output_stream=sys.stderr, sep='')
                        return
                    num_chunks_produced = 0
                    time_begin = time.perf_counter()
                    for c in chunker(clean_and_noise, target_sr, (new_uttid, *meta[1:])):
                        num_chunks_produced += 1
                        yield c
                    time_end = time.perf_counter()
                    chunk_stats.append(("snr-mixer", num_chunks_produced, time_end - time_begin))
        if verbosity > 3:
            tf_print(
                "chunk stats:\n{:>20s} {:>10s} {:>10s}\n".format("type", "chunks", "dur (s)"),
                "\n".join("{:>20s} {:>10d} {:>10.3f}".format(*stat) for stat in chunk_stats))

    if verbosity:
        tf_print("Using wav chunk loader, generating chunks of length {} with step size {} (milliseconds)".format(chunks["length_ms"], chunks["step_ms"]))
    return chunk_loader

def get_random_chunk_loader(paths, meta, wav_config, verbosity=0):
    raise NotImplementedError("todo")
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
            wav = load_wav(tf.constant(p, tf.string))
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
        tf_print("Using random wav chunk loader, drawing lengths (in frames) from", lengths, "with", overlap_ratio, "overlap ratio and", min_chunk_length, "minimum chunk length")
    return random_chunk_loader

# Use batch_size > 1 iff _every_ audio file in paths has the same amount of samples
# TODO: fix this mess
def extract_features_from_paths(feat_config, paths, meta, datagroup_key, verbosity=0):
    paths, meta, durations = list(paths), [m[:3] for m in meta], [m[3] for m in meta]
    assert len(paths) == len(meta) == len(durations), "Cannot extract features from paths when the amount of metadata {} and durations {} does not match the amount of wavfile paths {}".format(len(meta), len(durations), len(paths))
    wav_config = feat_config.get("wav_config")
    if wav_config:
        dataset_types = (
            (tf.float32, tf.int32),
            tf.string,
            tf.string)
        dataset_shapes = (
            (tf.TensorShape([None]), tf.TensorShape([])),
            tf.TensorShape([]),
            tf.TensorShape([]))
        if "chunks" in wav_config:
            expected_samples_per_file = wav_config.get("expected_samples_per_file", 1)
            if verbosity:
                print("Loading samples from wavs as fixed sized chunks, expecting up to {} consecutive elements from each file".format(expected_samples_per_file))
            chunk_loader_fn = get_chunk_loader(wav_config, verbosity, datagroup_key)
            def ds_generator(*args):
                return tf.data.Dataset.from_generator(
                    chunk_loader_fn,
                    dataset_types,
                    dataset_shapes,
                    args=args)
            min_duration = 1e-3 * wav_config["chunks"]["length_ms"]
            def has_min_chunk_length(path, meta, duration):
                ok = duration >= min_duration
                if verbosity and not ok:
                    tf_print("dropping too short (", duration, " sec < chunk len ", min_duration, " sec) file ", path, sep='', output_stream=sys.stderr)
                return ok
            def load_wav_with_meta(path, meta, duration):
                wav = load_wav(path)
                return wav.audio, wav.sample_rate, path, meta
            target_sr = wav_config.get("filter_sample_rate", -1)
            if verbosity:
                if target_sr < 0:
                    print("filter_sample_rate not defined in wav_config, wav files will not be filtered")
                else:
                    print("filter_sample_rate set to {}, wav files with mismatching sample rates will be ignored and no features will be extracted from those files".format(target_sr))
            def has_target_sample_rate(audio, sample_rate, path, meta):
                ok = target_sr > -1 and sample_rate == target_sr
                if verbosity and not ok:
                    tf_print("dropping wav due to wrong sample rate ", sample_rate, ", expected is ", target_sr, " file is ", path, sep='', output_stream=sys.stderr)
                return ok
            paths_t = tf.constant(paths, tf.string)
            meta_t = tf.constant(meta, tf.string)
            duration_t = tf.constant(durations, tf.float32)
            wavs = (tf.data.Dataset
                    .from_tensor_slices((paths_t, meta_t, duration_t))
                    .filter(has_min_chunk_length)
                    .map(load_wav_with_meta, num_parallel_calls=TF_AUTOTUNE)
                    .filter(has_target_sample_rate)
                    .interleave(
                        ds_generator,
                        block_length=expected_samples_per_file,
                        num_parallel_calls=TF_AUTOTUNE))
        else:
            print("unknown, non-empty wav_config given:")
            yaml_pprint(wav_config)
            raise NotImplementedError
        wavs = wavs.map(lambda wav, *meta: (audio_feat.Wav(wav[0], wav[1]), *meta))
    else:
        wav_paths = tf.data.Dataset.from_tensor_slices((
            tf.constant(paths, dtype=tf.string),
            tf.constant(meta, dtype=tf.string)))
        load_wav_with_meta = lambda path, *meta: (load_wav(path), *meta)
        wavs = wav_paths.map(load_wav_with_meta, num_parallel_calls=TF_AUTOTUNE)
    if "batch_wavs_by_length" in feat_config:
        window_size = feat_config["batch_wavs_by_length"]["max_batch_size"]
        if verbosity:
            print("Batching all wavs by equal length into batches of max size {}".format(window_size))
        key_fn = lambda wav, *meta: tf.cast(tf.size(wav.audio), tf.int64)
        reduce_fn = lambda key, group_ds: group_ds.batch(window_size)
        group_by_wav_length = tf.data.experimental.group_by_window(key_fn, reduce_fn, window_size)
        wavs_batched = wavs.apply(group_by_wav_length)
    else:
        batch_size = feat_config.get("batch_size", 1)
        if verbosity:
            print("Batching wavs with batch size", batch_size)
        wavs_batched = wavs.batch(batch_size)
    if verbosity:
        print("Applying feature extractor to batched wavs")
    feat_extract_args = feat_extraction_args_as_list(feat_config)
    # This function expects batches of wavs
    extract_feats = lambda wavs, *meta: (
        extract_features(wavs, *feat_extract_args),
        (*meta, wavs)
    )
    features = wavs_batched.map(extract_feats, num_parallel_calls=TF_AUTOTUNE)
    if "mean_var_norm_numpy" in feat_config:
        window_len = tf.constant(feat_config["mean_var_norm_numpy"]["window_len"], tf.int32)
        normalize_variance = tf.constant(feat_config["mean_var_norm_numpy"].get("normalize_variance", True), tf.bool)
        if verbosity:
            tf_print("Using numpy to apply mean_var_norm sliding window of length", window_len, "without padding. Will also normalize variance:", normalize_variance)
        def apply_mean_var_norm_numpy(feats, *rest):
            normalized = tf.numpy_function(
                mean_var_norm_nopad_slide_numpy,
                [feats, window_len, normalize_variance],
                feats.dtype)
            normalized.set_shape(feats.shape.as_list())
            return (normalized, *rest)
        features = features.map(apply_mean_var_norm_numpy, num_parallel_calls=TF_AUTOTUNE)
    return features

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

def parse_kaldi_features(utterance_list, features_path, utt2label, expected_shape, feat_conf):
    utt2feats = kaldiio.load_scp(features_path)
    normalize_mean_axis = feat_conf.pop("normalize_mean_axis", None)
    normalize_stddev_axis = feat_conf.pop("normalize_stddev_axis", None)
    assert not feat_conf, "feat_conf contains unrecognized keys: {}".format(','.join(str(k) for k in feat_conf))
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
            assert_shape(feats.shape)
            normalized = np.array(feats)
            if normalize_mean_axis is not None:
                mean = tf.math.reduce_mean(feats, axis=normalize_mean_axis, keepdims=True)
                normalized = feats - mean
            if normalize_stddev_axis is not None:
                stddev = tf.math.reduce_std(feats, axis=normalize_stddev_axis, keepdims=True)
                normalized = tf.math.divide_no_nan(normalized, stddev)
            yield normalized, (utt, utt2label[utt])
    return tf.data.Dataset.from_generator(
        datagen,
        (tf.float32, tf.string),
        (tf.TensorShape(expected_shape), tf.TensorShape([2])),
    )
