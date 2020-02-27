import contextlib
import collections
import io
import os
import random
import sys
import time
import wave

from . import audio_feat
from lidbox import yaml_pprint
import kaldiio
import librosa.core
import matplotlib.cm
import numpy as np
import soundfile
import tensorflow as tf
import webrtcvad

debug = False
if debug:
    TF_AUTOTUNE = None
    tf.autograph.set_verbosity(10, alsologtostdout=True)
else:
    TF_AUTOTUNE = tf.data.experimental.AUTOTUNE


def tf_print(*args, **kwargs):
    if "summarize" not in kwargs:
        kwargs["summarize"] = -1
    if "output_stream" not in kwargs:
        kwargs["output_stream"] = sys.stdout
    return tf.print(*args, **kwargs)

@tf.function
def feature_scaling(X, min, max, axis=None):
    """Apply feature scaling on X over given axis such that all values are between [min, max]"""
    X_min = tf.math.reduce_min(X, axis=axis, keepdims=True)
    X_max = tf.math.reduce_max(X, axis=axis, keepdims=True)
    return min + (max - min) * tf.math.divide_no_nan(X - X_min, X_max - X_min)

@tf.function
def cmvn_slide(X, window_len=300, normalize_variance=True):
    """Apply cepstral mean and variance normalization on batches of features matrices X with a given cmvn window length."""
    tf.debugging.assert_rank(X, 3, message="Input to cmvn_slide should be of shape (Batch, Timedim, Coefs)")
    if tf.shape(X)[1] <= window_len:
        # All frames of X fit inside one window, no need for sliding cmvn
        centered = X - tf.math.reduce_mean(X, axis=1, keepdims=True)
        if normalize_variance:
            return tf.math.divide_no_nan(centered, tf.math.reduce_std(X, axis=1, keepdims=True))
        else:
            return centered
    else:
        # Padding by reflecting the coefs along the time dimension should not dilute the means and variances as much as zeros would
        padding = tf.constant([[0, 0], [window_len//2, window_len//2 - 1 + (window_len&1)], [0, 0]])
        X_padded = tf.pad(X, padding, mode="REFLECT")
        cmvn_windows = tf.signal.frame(X_padded, window_len, 1, axis=1)
        tf.debugging.assert_equal(tf.shape(cmvn_windows)[1], tf.shape(X)[1], message="Mismatching amount of CMVN output windows and time steps in the input")
        centered = X - tf.math.reduce_mean(cmvn_windows, axis=2)
        if normalize_variance:
            return tf.math.divide_no_nan(centered, tf.math.reduce_std(cmvn_windows, axis=2))
        else:
            return centered

# NOTE tensorflow does not yet support non-zero axes in tf.gather when indices are ragged
# @tf.function
# def cmvn_gather(X, window_len=300):
#     """Same as cmvn_slide but without padding."""
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

def cmvn_nopad_slide_numpy(X, window_len, normalize_variance):
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
def extract_features(signals, feattype, spec_kwargs, melspec_kwargs, mfcc_kwargs, db_spec_kwargs, feat_scale_kwargs, cmvn_kwargs):
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
    if cmvn_kwargs:
        feat = cmvn_slide(feat, **cmvn_kwargs)
        tf.debugging.assert_all_finite(feat, "cmvn failed")
    return feat

def feat_extraction_args_as_list(feat_config):
    args = [feat_config["type"]]
    kwarg_dicts = [
        feat_config.get("spectrogram", {}),
        feat_config.get("melspectrogram", {}),
        feat_config.get("mfcc", {}),
        feat_config.get("db_spectrogram", {}),
        feat_config.get("sample_minmax_scaling", {}),
        feat_config.get("cmvn", {}),
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
        ds = ds.batch(config["batch_size"])
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
def attach_dataset_logger(ds, features_name, max_outputs=10, image_resize_kwargs=None, colormap="viridis", debug_squeeze_last_dim=False, num_batches=-1):
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
        if debug_squeeze_last_dim:
            samples = tf.squeeze(samples, -1)
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
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = np.sqrt((noise**2).mean())
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech

def get_chunk_loader(paths, meta_list, wav_config, verbosity, datagroup_key):
    chunks = wav_config["chunks"]
    target_sr = wav_config.get("target_sample_rate")
    augment_config = wav_config.get("augmentation", [])
    if datagroup_key != "train":
        if verbosity:
            print("skipping augmentation due to non-training datagroup: '{}'".format(datagroup_key))
        augment_config = []
    vad_trim = wav_config.get("webrtcvad_trim")
    for conf in augment_config:
        # prepare noise augmentation
        if conf["type"] == "additive_noise":
            noise_source_dir = conf["noise_source"]
            conf["noise_source"] = {}
            with open(os.path.join(noise_source_dir, "id2label")) as f:
                id2label = dict(l.strip().split() for l in f)
            label2path = collections.defaultdict(list)
            with open(os.path.join(noise_source_dir, "id2path")) as f:
                for id, path in (l.strip().split() for l in f):
                    label2path[id2label[id]].append(path)
            for noise_type, noise_paths in label2path.items():
                conf["noise_source"][noise_type] = noise_paths
    def trim_silence(signal, sr):
        vad_frame_ms = vad_trim["frame_ms"]
        assert vad_frame_ms in (10, 20, 30)
        assert sr == target_sr, "unexpected sample rate {}, cannot do vad_trim because wav write would distort pitch".format(sr)
        with contextlib.closing(io.BytesIO()) as buf:
            soundfile.write(buf, signal, sr, format="WAV")
            buf.seek(0)
            pcm_data, sr = read_wave(buf)
        assert sr == target_sr, "vad_trim failed during WAV read"
        step = int(sr * 1e-3 * vad_frame_ms * 2)
        if signal.size < step//2:
            return np.zeros(0, dtype=signal.dtype)
        frames = librosa.util.frame(signal, step//2, step//2, axis=0)
        vad_decisions = np.ones(frames.shape[0], dtype=np.bool)
        pcm_data_end = len(pcm_data) - len(pcm_data) % step
        a, b = len(range(0, pcm_data_end, step)), len(range(pcm_data_end, 0, -step))
        assert a == b == len(vad_decisions), (a, b, len(vad_decisions))
        del a, b
        vad = webrtcvad.Vad(vad_trim["aggressiveness"])
        for f, i in enumerate(range(0, pcm_data_end, step)):
            if not vad.is_speech(pcm_data[i:i+step], sr):
                vad_decisions[f] = False
            else:
                break
        for f, i in enumerate(range(pcm_data_end, 0, -step)):
            if not vad.is_speech(pcm_data[i-step:i], sr):
                vad_decisions[f] = False
            else:
                break
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
                yield (chunk, sr), meta[0] + "-{:06d}".format(i), meta[1]
    def chunk_loader():
        for p, meta in zip(paths, meta_list):
            utt, label, dataset = meta[:3]
            original_signal, sr = librosa.core.load(p, sr=target_sr, mono=True)
            if vad_trim:
                original_signal = trim_silence(original_signal, sr)
            if original_signal.size < int(sr * 1e-3 * chunks["length_ms"]):
                continue
            yield from chunker(original_signal, target_sr, meta)
            for conf in augment_config:
                if "datasets_include" in conf and dataset not in conf["datasets_include"]:
                    continue
                if "datasets_exclude" in conf and dataset in conf["datasets_exclude"]:
                    continue
                if conf["type"] == "speed_modification":
                    # apply naive speed modification by resampling
                    for rate in conf["range"]:
                        if rate == 1:
                            signal = original_signal
                        else:
                            with contextlib.closing(io.BytesIO()) as buf:
                                soundfile.write(buf, original_signal, int(rate * target_sr), format="WAV")
                                buf.seek(0)
                                signal, _ = librosa.core.load(buf, sr=target_sr, mono=True)
                        new_uttid = "{:s}-speed{:.3f}".format(utt, rate)
                        yield from chunker(signal, target_sr, (new_uttid, *meta[1:]))
                elif conf["type"] == "additive_noise":
                    for noise_type, db_min, db_max in conf["snr-def"]:
                        noise_signal = np.zeros(0, dtype=original_signal.dtype)
                        while noise_signal.size < original_signal.size:
                            rand_noise_path = random.choice(conf["noise_source"][noise_type])
                            sig, _ = librosa.core.load(rand_noise_path, sr=target_sr, mono=True)
                            noise_signal = np.concatenate((noise_signal, sig))
                        noise_begin = random.randint(0, noise_signal.size - original_signal.size)
                        noise_signal = noise_signal[noise_begin:noise_begin+original_signal.size]
                        snr_db = random.randint(db_min, db_max)
                        clean, noise, clean_and_noise = snr_mixer(original_signal, noise_signal, snr_db)
                        new_uttid = "{:s}-{:s}_snr{:d}".format(utt, noise_type, snr_db)
                        yield from chunker(clean_and_noise, target_sr, (new_uttid, *meta[1:]))
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
                    print("skipping file", p, ", it has a sample rate", wav.sample_rate, "but config has 'filter_sample_rate'", wav_config["filter_sample_rate"])
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
def extract_features_from_paths(feat_config, paths, meta, datagroup_key, trim_audio=None, debug_squeeze_last_dim=False, verbosity=0):
    paths, meta = list(paths), list(meta)
    assert len(paths) == len(meta), "Cannot extract features from paths when the amount of metadata {} does not match the amount of wavfile paths {}".format(len(meta), len(paths))
    wav_config = feat_config.get("wav_config")
    if wav_config:
        dataset_types = (tf.float32, tf.int32), tf.string, tf.string
        dataset_shapes = (tf.TensorShape([None]), tf.TensorShape([])), tf.TensorShape([]), tf.TensorShape([])
        if "random_chunks" in wav_config:
            # TODO interleave or without generator might improve performance
            wavs = tf.data.Dataset.from_generator(
                get_random_chunk_loader(paths, meta, wav_config, verbosity),
                dataset_types,
                dataset_shapes)
        if "chunks" in wav_config:
            wavs = tf.data.Dataset.from_generator(
                get_chunk_loader(paths, meta, wav_config, verbosity, datagroup_key),
                dataset_types,
                dataset_shapes)
        else:
            print("unknown, non-empty wav_config given:")
            yaml_pprint(wav_config)
            assert False
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
    if "cmvn_numpy" in feat_config:
        window_len = tf.constant(feat_config["cmvn_numpy"]["window_len"], tf.int32)
        normalize_variance = tf.constant(feat_config["cmvn_numpy"].get("normalize_variance", True), tf.bool)
        if verbosity:
            tf_print("Using numpy to apply cmvn sliding window of length", window_len, "without padding. Will also normalize variance:", normalize_variance)
        def apply_cmvn_numpy(feats, *rest):
            normalized = tf.numpy_function(
                cmvn_nopad_slide_numpy,
                [feats, window_len, normalize_variance],
                feats.dtype)
            normalized.set_shape(feats.shape.as_list())
            return (normalized, *rest)
        features = features.map(apply_cmvn_numpy, num_parallel_calls=TF_AUTOTUNE)
    features = features.unbatch()
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
