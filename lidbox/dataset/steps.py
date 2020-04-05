import collections
import logging
import os
import random
import time

random.seed(42)
logger = logging.getLogger("dataset")

import tensorflow as tf

import lidbox
import lidbox.dataset.tf_utils as tf_utils
import lidbox.features as features
import lidbox.features.audio as audio_features


if lidbox.DEBUG:
    TF_AUTOTUNE = None
    tf.debugging.set_log_device_placement(True)
else:
    TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

Step = collections.namedtuple("Step", ("key", "kwargs"))

def _read_and_split(path, num_columns):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                yield l.split(' ', num_columns)[:num_columns]

def _feature_extraction_kwargs_to_args(config):
    valid_args = [
        "type",
        "spectrogram",
        "melspectrogram",
        "mfcc",
        "db_spectrogram",
        "sample_minmax_scaling",
        "window_normalization",
    ]
    return [config.get(arg, {}) for arg in valid_args]

def _element_shapes_dict(x):
    return {k: list(tf.shape(v).numpy()) for k, v in x.items()}

def _pretty_dict(d):
    return "\n  ".join("{}: {}".format(k, p) for k, p in d.items())


def apply_filters(ds, config):
    logger.debug("Applying filters on every element in the dataset, keeping only elements which match the given config:\n  %s", _pretty_dict(config))
    filter_sample_rate = tf.constant(config.get("sample_rate", -1), tf.int32)
    filter_min_signal_length_sec = tf.constant(config.get("min_signal_length_sec", 0) * 1e-3, tf.float32)
    def all_ok(x):
        if "sample_rate" in x and filter_sample_rate != -1 and x["sample_rate"] != filter_sample_rate:
            return False
        if "signal" in x and tf.size(x["signal"]) < tf.cast(tf.cast(x["sample_rate"], tf.float32) * filter_min_signal_length_sec, tf.int32):
            return False
        return True
    return ds.filter(all_ok)


def apply_vad(ds):
    logger.debug("Applying previously computed voice activity decisions.")
    def filter_signals_by_vad_decisions(x):
        vad_frame_length_sec = 1e-3 * tf.cast(x["vad_frame_length_ms"], tf.float32)
        vad_frame_length = tf.cast(tf.cast(x["sample_rate"], tf.float32) * vad_frame_length_sec, tf.int32)
        frames = tf.signal.frame(x["signal"], vad_frame_length, vad_frame_length, axis=0)
        voiced_signal = tf.reshape(frames[x["vad_is_speech"]], [-1])
        return dict(x, signal=voiced_signal)
    return ds.map(filter_signals_by_vad_decisions, num_parallel_calls=TF_AUTOTUNE)


def as_supervised(ds):
    logger.debug("Converting all elements to tuple pairs (inputs, targets) and dropping all other values.")
    def _as_supervised(x):
        return x["input"], x["target"]
    return ds.map(_as_supervised, num_parallel_calls=TF_AUTOTUNE)


def cache(ds, cache_dir=None, cache_key=None, batch_size=1):
    if cache_dir is None:
        logger.warning("Caching dataset in batches of size %d into memory.", batch_size)
        cache_file = ''
    else:
        if cache_key is None:
            cache_key = str(int(time.time()))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, cache_key)
        logger.info("Caching dataset in batches of size %d to directory '%s' with key '%s'.", batch_size, cache_dir, cache_key)
    return (ds.batch(batch_size)
              .prefetch(TF_AUTOTUNE)
              .cache(cache_file)
              .prefetch(TF_AUTOTUNE)
              .unbatch())


def compute_vad(ds, aggressiveness, vad_frame_length_ms, min_non_speech_length_ms):
    vad_frame_length_sec = tf.constant(vad_frame_length_ms * 1e-3, tf.float32)
    min_non_speech_frames = tf.constant(min_non_speech_length_ms // vad_frame_length_ms, tf.int32)
    logger.debug("Computing voice activity detection decisions on %d ms long windows.\nMinimum length of continous non-speech segment before it is marked as non-speech is %d ms.", vad_frame_length_ms, min_non_speech_length_ms)
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
        return dict(x, vad_is_speech=vad_decisions, vad_frame_length_ms=vad_frame_length_ms)
    return ds.map(append_vad_decisions, num_parallel_calls=TF_AUTOTUNE)


def consume(ds, log_interval=-1):
    speed = 0
    last_update = 0
    counter = time.perf_counter()
    def counter_step(i):
        nonlocal speed, last_update, counter
        stop = time.perf_counter()
        speed = max(0, (i - last_update) / (stop - counter))
        logger.info("%d done, %.3f elements per second.", i, speed)
        last_update = i
        counter = time.perf_counter()
    for i, element in ds.enumerate(start=1).as_numpy_iterator():
        if log_interval > -1 and i % log_interval == 0:
            counter_step(i)
    counter_step(i)
    return ds


def consume_to_tensorboard(ds, summary_dir, config):
    colors = tf_utils.matplotlib_colormap_to_tensor(config.get("colormap", "viridis"))
    image_size_multiplier = tf.constant(config.get("image_size_multiplier", 1), tf.float32)
    batch_size = tf.constant(config["batch_size"], tf.int64)
    max_outputs = tf.constant(config.get("max_elements_per_batch", batch_size.numpy()), tf.int64)
    num_batches = tf.constant(config.get("num_batches", -1), tf.int64)

    @tf.function
    def inspect_batches(batch_idx, batch):
        tf.debugging.assert_greater([tf.size(v) for v in batch.values()], 0, message="Empty batch given to tensorboard logger.")
        inputs = batch["input"][:max_outputs]
        targets = batch["target"][:max_outputs]
        tf.summary.histogram("inputs", inputs, step=batch_idx)
        tf.summary.histogram("targets", targets, step=batch_idx)
        images = tf_utils.tensors_to_rgb_images(inputs, colors, image_size_multiplier)
        tf.summary.image("inputs/img", images, step=batch_idx, max_outputs=max_outputs)
        if "signal" in batch and tf.size(batch["signal"]) > 0:
            sample_rates = batch["sample_rate"][:max_outputs]
            tf.debugging.assert_equal(sample_rates, [sample_rates[0]], message="Unable to add audio to tensorboard summary due to signals in the batch having different sample rates")
            signals = tf.expand_dims(batch["signal"][:max_outputs], -1)
            tf.summary.audio("utterances", signals, sample_rates[0], step=batch_idx, encoding="wav")
        enumerated_uttids = tf.strings.reduce_join(
                (tf.strings.as_string(tf.range(1, max_outputs + 1)), batch["id"][:max_outputs]),
                axis=0,
                separator=": ")
        tf.summary.text("utterance_ids", enumerated_uttids, step=batch_idx)
        return batch

    logger.debug("Writing Tensorboard data into '%s'", summary_dir)
    writer = tf.summary.create_file_writer(summary_dir)
    with writer.as_default():
        _ = consume(ds.batch(batch_size)
                      .take(num_batches)
                      .enumerate()
                      .map(inspect_batches, num_parallel_calls=TF_AUTOTUNE))
    return ds


def create_signal_chunks(ds, length_ms, step_ms, max_pad_ms=0, deterministic_output_order=True, max_num_chunks_per_signal=int(1e6), avg_num_chunks_from_signals=100):
    logger.debug("Dividing every signal in the dataset into new signals by creating signal chunks of length %d ms and offset %d ms. Maximum amount of padding allowed in the last chunk is %d ms.", length_ms, step_ms, max_pad_ms)
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


def drop_empty(ds):
    non_scalar_keys = ("signal", "input")
    logger.debug("Dropping every element which have an empty tensor at any of the non-scalar element keys:\n  %s", "\n  ".join(non_scalar_keys))
    def is_not_empty(x):
        empty = False
        for k in non_scalar_keys:
            if k in x and tf.size(x[k]) == 0:
                empty = True
        return not empty
    return ds.filter(is_not_empty)


def drop_keys_in_set(ds, keys):
    logger.debug("For each element in the dataset, dropping values with keys: %s.", ', '.join(keys))
    def drop_keys(x):
        return {k: v for k, v in x.items() if k not in keys}
    return ds.map(drop_keys, num_parallel_calls=TF_AUTOTUNE)


def extract_features(ds, config):
    feature_type = tf.constant(config["type"], tf.string)
    args = _feature_extraction_kwargs_to_args(config)
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if "device" in config:
        tf_device = config["device"]
    elif gpu_devices:
        tf_device = "/GPU:0"
    else:
        tf_device = "/CPU:0"
    logger.debug("Extracting '%s' features on device '%s' with arguments:\n  %s", config["type"], tf_device, "\n  ".join(repr(a) for a in args[1:]))
    batch_size = tf.constant(config.get("batch_size", 1), tf.int64)
    logger.debug("Batching signals with batch size %s, extracting features in batches.", batch_size.numpy())
    def append_features(x):
        with tf.device(tf_device):
            features = tf_utils.extract_features(x["signal"], x["sample_rate"], *args)
        feature_type_batch = tf.repeat(feature_type, tf.shape(features)[0])
        return dict(x, input=features, feature_type=feature_type_batch)
    return (ds.batch(batch_size)
              .map(append_features, num_parallel_calls=TF_AUTOTUNE)
              .unbatch())


def filter_keys_in_set(ds, keys):
    logger.debug("For each element in the dataset, keeping only values with keys: %s.", ', '.join(keys))
    def filter_keys(x):
        return {k: v for k, v in x.items() if k in keys}
    return ds.map(filter_keys, num_parallel_calls=TF_AUTOTUNE)


def initialize(ds, labels, metadata_paths, file_limit=-1, shuffle_files=False):
    if ds is not None:
        logger.warning("Step 'initialize' is being applied on an already initialized dataset, all state will be lost.")
    ds = None
    if logger.getEffectiveLevel() == logging.DEBUG:
        msg = "Initializing dataset from {} metadata paths:\n  {}".format(len(metadata_paths), _pretty_dict(collections.OrderedDict(metadata_paths)))
        logger.debug(msg)
    valid_paths = []
    for k, p in metadata_paths:
        if os.path.exists(p):
            valid_paths.append((k, p))
        else:
            logger.warning("Dropping non-existing '%s' metadata file '%s'.", k, p)
    metadata_paths = valid_paths
    del valid_paths
    key2meta = {key: dict(_read_and_split(path, 2)) for key, path in metadata_paths}
    if len(key2meta) < len(metadata_paths):
        logger.warning("Possible duplicate metadata keys, %d paths were given but only %d unique metadata files were parsed.", len(metadata_paths), len(key2meta))
    # Get all utterance ids from some metadata dict
    all_ids = set(key2meta[list(key2meta)[0]].keys())
    if not all(set(v.keys()) == all_ids for v in key2meta.values()):
        logger.critical("Mismatching amount of utterance ids in meta files, unable to deduce the set of all utterance ids that should be used. Make sure all metadata files contain exactly the same set of utterance ids.")
        return
    all_ids = sorted(all_ids)
    if shuffle_files:
        random.shuffle(all_ids)
    all_ids = all_ids[:file_limit]
    init_data = {"id": tf.constant(all_ids, tf.string)}
    for key, meta in key2meta.items():
        init_data[key] = tf.constant([meta[i] for i in all_ids], tf.string)
    # Peek one path to do a sanity check
    if "path" in init_data and tf.size(init_data["path"]).numpy() and not os.path.exists(init_data["path"][0].numpy().decode("utf-8")):
        logger.error("First path '%s' does not exist, check that all paths are valid before continuing", init_data["path"][0].numpy().decode("utf-8"))
        return
    ds = tf.data.Dataset.from_tensor_slices(init_data)
    logger.debug("Generating label2target lookup table from indexes of array:\n  %s", '\n  '.join(labels))
    label2int, _ = tf_utils.make_label2onehot(tf.constant(labels, tf.string))
    append_labels_as_targets = lambda x: dict(x, target=label2int.lookup(x["label"]))
    return ds.map(append_labels_as_targets, num_parallel_calls=TF_AUTOTUNE)


def load_audio(ds):
    logger.debug("Reading audio files from the path of each element and appending the read signals and their sample rates to each element.")
    def append_signals(x):
        signal, sample_rate = audio_features.read_wav(x["path"])
        return dict(x, signal=signal, sample_rate=sample_rate)
    return ds.map(append_signals, num_parallel_calls=TF_AUTOTUNE)


def reduce_stats(ds, statistic):
    logger.debug("Iterating over whole dataset to compute statistic '%s'", statistic)
    if statistic == "num_elements":
        num_elements = ds.reduce(0, lambda c, x: c + 1)
        logger.info("Num elements: %d.", int(num_elements.numpy()))
    else:
        logger.error("Unknown statistic type '%s', cannot compute stats for dataset.", statistic)
    return ds


def show_all_elements(ds):
    for i, element in ds.enumerate(start=1).as_numpy_iterator():
        if not isinstance(element, dict):
            element = {i: v for i, v in enumerate(element)}
        logger.info("Element %d:\nshapes: %s\ncontents: %s.", i, repr(_element_shapes_dict(element)), repr(element))
    return ds


VALID_STEP_FUNCTIONS = {
    "apply_filters": apply_filters,
    "apply_vad": apply_vad,
    "as_supervised": as_supervised,
    "cache": cache,
    "compute_vad": compute_vad,
    "consume": consume,
    "consume_to_tensorboard": consume_to_tensorboard,
    "create_signal_chunks": create_signal_chunks,
    "drop_empty": drop_empty,
    "drop_keys_in_set": drop_keys_in_set,
    "extract_features": extract_features,
    "filter_keys_in_set": filter_keys_in_set,
    "initialize": initialize,
    "load_audio": load_audio,
    "reduce_stats": reduce_stats,
    "show_all_elements": show_all_elements,
}
