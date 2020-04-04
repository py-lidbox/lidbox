import collections
import logging
import os
import random
import time

random.seed(42)
logger = logging.getLogger("dataset")

import tensorflow as tf

import lidbox
import lidbox.features.audio as audio_features
import lidbox.dataset.tf_util

TF_AUTOTUNE = None if lidbox.DEBUG else tf.data.experimental.AUTOTUNE


Step = collections.namedtuple("Step", ("key", "kwargs"))


def _read_and_split(path, num_columns):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                yield l.split(' ', num_columns)[:num_columns]

def _feat_extraction_args_as_list(config):
    args = [config["type"]]
    valid_args = [
        "spectrogram",
        "melspectrogram",
        "mfcc",
        "db_spectrogram",
        "sample_minmax_scaling",
        "window_normalization",
    ]
    args.extend([config.get(arg, {}) for arg in valid_args])
    return args

def _element_shapes_dict(x):
    return {k: list(tf.shape(v).numpy()) for k, v in x.items()}


def consume(ds, log_interval=-1):
    speed = 0
    last_update = 0
    counter = time.perf_counter()
    def counter_step(i):
        nonlocal speed, last_update, counter
        stop = time.perf_counter()
        speed = max(0, (i - last_update) / (stop - counter))
        logger.info("%d elements consumed at %.3f elements per second", i, speed)
        last_update = i
        counter = time.perf_counter()
    for i, element in ds.enumerate(start=1).as_numpy_iterator():
        if log_interval > -1 and i % log_interval == 0:
            counter_step(i)
    counter_step(i)
    return ds


def show_all_elements(ds):
    if logger.getEffectiveLevel() != logging.DEBUG:
        logger.warning("Skipping show_all_elements without debug verbosity. This step will create a lot of output and is only enabled in debug mode.")
    else:
        for i, element in ds.enumerate(start=1).as_numpy_iterator():
            logger.debug("Element %d:\nshapes: %s\ncontents: %s", i, repr(_element_shapes_dict(element)), repr(element))
    return ds


def initialize(ds, metadata_paths, file_limit=-1, shuffle_files=False):
    if ds is not None:
        logger.warning("Step 'initialize' is being applied on an already initialized dataset, all state will be lost")
    ds = None
    if logger.getEffectiveLevel() == logging.DEBUG:
        msg = "Initializing dataset from {} metadata paths:\n  {}".format(len(metadata_paths), "\n".join("  {}: {}".format(key, path) for key, path in metadata_paths))
        logger.debug(msg)
    valid_paths = []
    for k, p in metadata_paths:
        if os.path.exists(p):
            valid_paths.append((k, p))
        else:
            logger.warning("Dropping non-existing '%s' metadata file '%s'", k, p)
    metadata_paths = valid_paths
    del valid_paths
    key2meta = {key: dict(_read_and_split(path, 2)) for key, path in metadata_paths}
    if len(key2meta) < len(metadata_paths):
        logger.warning("Possible duplicate metadata keys, %d paths were given but only %d unique metadata files were parsed", len(metadata_paths), len(key2meta))
    # Get all utterance ids from some metadata dict
    all_ids = set(key2meta[list(key2meta)[0]].keys())
    if not all(set(v.keys()) == all_ids for v in key2meta.values()):
        logger.critical("Mismatching amount of utterance ids in meta files, make sure all metadata files contain exactly the same set of utterance ids")
        return
    all_ids = sorted(all_ids)
    if shuffle_files:
        random.shuffle(all_ids)
    all_ids = all_ids[:file_limit]
    init_data = {"id": tf.constant(all_ids, tf.string)}
    for key, meta in key2meta.items():
        init_data[key] = tf.constant([meta[i] for i in all_ids], tf.string)
    return tf.data.Dataset.from_tensor_slices(init_data)


def load_audio(ds):
    def append_signals(x):
        signal, sample_rate = audio_features.read_wav(x["path"])
        return dict(x, signal=signal, sample_rate=sample_rate)
    return ds.map(append_signals, num_parallel_calls=TF_AUTOTUNE)


def extract_features(ds, config):
    feature_type = tf.constant(config["type"], tf.string)
    args = _feat_extraction_args_as_list(config)
    logger.debug("Feature extraction args: %s", ' '.join(repr(a) for a in args))
    batch_size = config.get("batch_size", 1)
    logger.debug("Batching signals with batch size %s, extracting features in batches", batch_size)
    def append_features(x):
        features = lidbox.dataset.tf_util.extract_features(x["signal"], x["sample_rate"], *args)
        feature_type_batch = tf.repeat(feature_type, tf.shape(features)[0])
        return dict(x, features=features, feature_type=feature_type_batch)
    return (ds.batch(batch_size)
              .map(append_features, num_parallel_calls=TF_AUTOTUNE)
              .unbatch())


def compute_vad(ds, aggressiveness, vad_frame_length_ms, min_non_speech_length_ms):
    vad_frame_length_sec = tf.constant(vad_frame_length_ms * 1e-3, tf.float32)
    min_non_speech_frames = tf.constant(min_non_speech_length_ms // vad_frame_length_ms, tf.int32)
    def append_vad_decisions(x):
        signal, sample_rate = x["signal"], x["sample_rate"]
        vad_frame_length = tf.cast(tf.cast(sample_rate, tf.float32) * vad_frame_length_sec, tf.int32)
        frames = tf.signal.frame(signal, vad_frame_length, vad_frame_length, axis=0)
        _, pcm_data = audio_features.wav_to_pcm_data(signal, sample_rate)
        args = (signal,
                sample_rate,
                pcm_data,
                vad_frame_length,
                aggressiveness,
                min_non_speech_frames)
        vad_decisions = tf.numpy_function(audio_features.numpy_fn_get_webrtcvad_decisions, args, tf.bool)
        vad_decisions = tf.reshape(vad_decisions, [tf.shape(frames)[0]])
        return dict(x, vad_is_speech=vad_decisions)
    return ds.map(append_vad_decisions, num_parallel_calls=TF_AUTOTUNE)


def apply_vad(ds, vad_frame_length_ms, **kwargs):
    vad_frame_length_sec = tf.constant(vad_frame_length_ms * 1e-3, tf.float32)
    def filter_signals_by_vad_decisions(x):
        vad_frame_length = tf.cast(tf.cast(x["sample_rate"], tf.float32) * vad_frame_length_sec, tf.int32)
        frames = tf.signal.frame(x["signal"], vad_frame_length, vad_frame_length, axis=0)
        voiced_signal = tf.reshape(frames[x["vad_is_speech"]], [-1])
        return dict(x, signal=voiced_signal)
    return ds.map(filter_signals_by_vad_decisions, num_parallel_calls=TF_AUTOTUNE)


def apply_filters(ds, config):
    filter_sample_rate = tf.constant(config.get("sample_rate", -1), tf.int32)
    filter_min_signal_length_sec = tf.constant(config.get("min_signal_length_sec", 0) * 1e-3, tf.float32)
    def all_ok(x):
        if "sample_rate" in x and filter_sample_rate != -1 and x["sample_rate"] != filter_sample_rate:
            return False
        if "signal" in x and tf.size(x["signal"]) < tf.cast(tf.cast(x["sample_rate"], tf.float32) * filter_min_signal_length_sec, tf.int32):
            return False
        return True
    return ds.filter(all_ok)


def create_signal_chunks(ds, length_ms, step_ms, max_pad_ms=0, deterministic_output_order=True, max_num_chunks_per_signal=int(1e6), avg_num_chunks_from_signals=100):
    chunk_length_sec = tf.constant(1e-3 * length_ms, tf.float32)
    chunk_step_sec = tf.constant(1e-3 * step_ms, tf.float32)
    max_pad_sec = tf.constant(1e-3 * max_pad_ms, tf.float32)
    id_str_padding = tf.cast(tf.round(audio_features.log10(tf.cast(max_num_chunks_per_signal, tf.float32))), tf.int32)
    def chunks_to_elements(chunk, chunk_num, x):
        chunk_num_str = tf.strings.as_string(chunk_num, width=id_str_padding, fill='0')
        chunk_id = tf.strings.join((x["id"], chunk_num_str), separator='-')
        out = dict(x, signal=tf.reshape(chunk, [-1]), id=chunk_id)
        if "duration" in x:
            duration_str = tf.strings.as_string(tf.cast(x["sample_rate"] * tf.size(chunk), tf.float32))
            out = dict(out, duration=duration_str)
        return out
    def chunk_signal_and_flatten(x):
        signal = x["signal"]
        sample_rate = tf.cast(x["sample_rate"], tf.float32)
        chunk_length = tf.cast(sample_rate * chunk_length_sec, tf.int32)
        chunk_step = tf.cast(sample_rate * chunk_step_sec, tf.int32)
        max_pad = tf.cast(sample_rate * max_pad_sec, tf.int32)
        num_full_chunks = tf.math.maximum(0, 1 + (tf.size(signal) - chunk_length) // chunk_step)
        tf.debugging.assert_less(num_full_chunks, max_num_chunks_per_signal, message="Too many chunks created from signal, cannot create unique utterance ids, raise the max_num_chunks_per_signal parameter")
        last_chunk_length = tf.size(signal) - num_full_chunks * chunk_step
        if last_chunk_length < chunk_length and chunk_length <= last_chunk_length + max_pad:
            signal = tf.pad(signal, [[0, chunk_length - last_chunk_length]])
        chunks = tf.signal.frame(signal, chunk_length, chunk_step, axis=0)
        num_chunks = tf.cast(tf.shape(chunks)[0], tf.int64)
        chunk_ds = tf.data.Dataset.from_tensor_slices(chunks)
        chunk_nums_ds = tf.data.Dataset.range(1, num_chunks + 1)
        repeat_x_ds = tf.data.Dataset.from_tensors(x).repeat(num_chunks)
        return (tf.data.Dataset
                  .zip((chunk_ds, chunk_nums_ds, repeat_x_ds))
                  .map(chunks_to_elements))
    return ds.interleave(
                chunk_signal_and_flatten,
                block_length=avg_num_chunks_from_signals,
                num_parallel_calls=TF_AUTOTUNE,
                deterministic=deterministic_output_order)


def drop_keys(ds, keys):
    def drop_keys(x):
        return {k: v for k, v in x.items() if k not in keys}
    return ds.map(drop_keys, num_parallel_calls=TF_AUTOTUNE)


def drop_empty(ds):
    def is_not_empty(x):
        empty = False
        for k in ("signal", "features"):
            if k in x and tf.size(x[k]) == 0:
                empty = True
        return not empty
    return ds.filter(is_not_empty)


def cache(ds, cache_dir=None, cache_name=None):
    if cache_dir is None:
        logger.warning("Caching dataset in memory")
        cache_file = None
    else:
        if cache_name is None:
            cache_name = str(int(time.time()))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, cache_name)
        logger.info("Caching dataset to '%s'", cache_file)
    return ds.cache(cache_file)


def reduce_stats(ds, statistic):
    if statistic == "num_elements":
        num_elements = ds.reduce(0, lambda c, x: c + 1)
        logger.info("Num elements: %d", int(num_elements.numpy()))
    else:
        logger.error("Unknown statistic type '%s', cannot compute stats for dataset", statistic)
    return ds


VALID_STEP_FUNCTIONS = {
    "apply_filters": apply_filters,
    "apply_vad": apply_vad,
    "cache": cache,
    "compute_vad": compute_vad,
    "consume": consume,
    "create_signal_chunks": create_signal_chunks,
    "drop_empty": drop_empty,
    "drop_keys": drop_keys,
    "extract_features": extract_features,
    "initialize": initialize,
    "load_audio": load_audio,
    "reduce_stats": reduce_stats,
    "show_all_elements": show_all_elements,
}
