import collections
import logging
import os
import random
import time

logger = logging.getLogger("dataset")

import tensorflow as tf
TF_VERSION_MAJOR, TF_VERSION_MINOR = tuple(int(x) for x in tf.version.VERSION.split(".")[:2])

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


def append_predictions(ds, predictions):
    """
    Add predictions to each element in ds.
    """
    def _append_predictions(x, p):
        return dict(x, prediction=p)
    predictions_ds = tf.data.Dataset.from_tensor_slices(predictions)
    return (tf.data.Dataset
              .zip((ds, predictions_ds))
              .map(_append_predictions, num_parallel_calls=TF_AUTOTUNE))


def apply_filters(ds, config):
    """
    Drop all elements from ds which do not satisfy all filter conditions given in config.
    """
    logger.debug("Applying filters on every element in the dataset, keeping only elements which match the given config:\n  %s", _pretty_dict(config))
    filter_sample_rate = tf.constant(config.get("sample_rate", -1), tf.int32)
    filter_min_signal_length_sec = tf.constant(config.get("min_signal_length_sec", 0) * 1e-3, tf.float32)
    # filter_min_input_shape = tf.constant(config.get("min_input_shape", []), tf.int32)
    def all_ok(x):
        if "sample_rate" in x and filter_sample_rate != -1 and x["sample_rate"] != filter_sample_rate:
            return False
        if "signal" in x and tf.size(x["signal"]) < tf.cast(tf.cast(x["sample_rate"], tf.float32) * filter_min_signal_length_sec, tf.int32):
            return False
        # if "input" in x and tf.size(filter_min_input_shape) != 0 and tf.math.reduce_any(tf.shape(x["input"]) < filter_min_input_shape):
            # return False
        return True
    return ds.filter(all_ok)


def apply_vad(ds):
    """
    Assuming each element of ds have voice activity detection decisions, use the decisions to drop non-speech frames.
    """
    logger.debug("Applying previously computed voice activity decisions.")
    def filter_signals_by_vad_decisions(x):
        vad_frame_length_sec = 1e-3 * tf.cast(x["vad_frame_length_ms"], tf.float32)
        vad_frame_length = tf.cast(tf.cast(x["sample_rate"], tf.float32) * vad_frame_length_sec, tf.int32)
        frames = tf.signal.frame(x["signal"], vad_frame_length, vad_frame_length, axis=0)
        voiced_signal = tf.reshape(frames[x["vad_is_speech"]], [-1])
        return dict(x, signal=voiced_signal)
    return ds.map(filter_signals_by_vad_decisions, num_parallel_calls=TF_AUTOTUNE)


def as_supervised(ds):
    """
    Convert all element dictionaries to tuples of (inputs, targets) pairs that can be given to a Keras model as input.
    """
    logger.debug("Converting all elements to tuple pairs (inputs, targets) and dropping all other values.")
    def _as_supervised(x):
        return x["input"], x["target"]
    return ds.map(_as_supervised, num_parallel_calls=TF_AUTOTUNE)


def cache(ds, directory=None, batch_size=1, cache_key=None):
    """
    Cache all elements of ds to disk or memory.
    """
    if directory is None:
        logger.warning("Caching dataset in batches of size %d into memory.", batch_size)
        cache_file = ''
    else:
        if cache_key is None:
            cache_key = str(int(time.time()))
        os.makedirs(directory, exist_ok=True)
        cache_file = os.path.join(directory, cache_key)
        if os.path.exists(cache_file + ".index"):
            logger.info("Loading elements from existing cache in directory '%s' with key '%s'.", directory, cache_key)
        else:
            logger.info("Caching dataset in batches of size %d to directory '%s' with key '%s'.", batch_size, directory, cache_key)
    return (ds.batch(batch_size)
              .prefetch(TF_AUTOTUNE)
              .cache(cache_file)
              .prefetch(TF_AUTOTUNE)
              .unbatch())


def compute_webrtc_vad(ds, aggressiveness, vad_frame_length_ms, min_non_speech_length_ms):
    """
    Compute voice activity detection with WebRTC VAD.
    """
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
    """
    Iterate over ds to exhaust the iterator and fully evaluate the preceding pipeline.
    """
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
    i = 0
    for i, element in ds.enumerate(start=1).as_numpy_iterator():
        if log_interval > -1 and i % log_interval == 0:
            counter_step(i)
    counter_step(i)
    return ds


def consume_to_tensorboard(ds, summary_dir, config, skip_if_exists=True):
    """
    Inspect the elements of ds by dumping them to TensorBoard summaries, e.g. features as spectrogram images and signals as wav audio.
    """
    if skip_if_exists and os.path.isdir(summary_dir) and any(p.name.startswith("events") for p in os.scandir(summary_dir) if p.is_file()):
        logger.info("Skipping TensorBoard step since 'skip_if_exists' is True and directory '%s' already contains tf event files", summary_dir)
        return ds
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
    """
    Divide the signals of each element of ds into fixed length chunks and create new utterances from the created chunks.
    The metadata of each element is repeated into into each chunk, except for the utterance ids, which will be appended by the chunk number.
    """
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
    interleave_kwargs = {
            "block_length": avg_num_chunks_from_signals,
            "num_parallel_calls": TF_AUTOTUNE,
            "deterministic": deterministic_output_order}
    if TF_VERSION_MAJOR == 2 and TF_VERSION_MINOR < 2:
        del interleave_kwargs["deterministic"]
        logger.warning("Deleted unsupported 'deterministic' kwarg from tf.data.Dataset.interleave call, TF version >= 2.2 is required.")
    return ds.interleave(chunk_signal_and_flatten, **interleave_kwargs)


def drop_empty(ds):
    """
    Drop all elements that contain an empty non-scalar value, e.g. signals of size 0 or spectrograms with 0 time frames.
    """
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
    """
    Given a set (or iterable) keys, delete those keys from every element of ds.
    """
    logger.debug("For each element in the dataset, dropping values with keys: %s.", ', '.join(keys))
    def drop_keys(x):
        return {k: v for k, v in x.items() if k not in keys}
    return ds.map(drop_keys, num_parallel_calls=TF_AUTOTUNE)


def extract_features(ds, config):
    """
    Extract features from signals of each element in ds and add them under 'input' key to each element.
    By default, feature extraction is requested to be placed on the first visible GPU, falling back on a CPU only if GPUs are not available.
    """
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
        feature_types = tf.repeat(feature_type, tf.shape(features)[0])
        return dict(x, input=features, feature_type=feature_types)
    return (ds.batch(batch_size)
              .prefetch(TF_AUTOTUNE)
              .map(append_features, num_parallel_calls=TF_AUTOTUNE)
              .unbatch())


def filter_keys_in_set(ds, keys):
    """
    Inverse of drop_keys_in_set.
    E.g. instead of dropping keys, keys are kept only if they are in the set 'keys'.
    """
    logger.debug("For each element in the dataset, keeping only values with keys: %s.", ', '.join(keys))
    def filter_keys(x):
        return {k: v for k, v in x.items() if k in keys}
    return ds.map(filter_keys, num_parallel_calls=TF_AUTOTUNE)


def group_by_sequence_length(ds, max_batch_size, min_batch_size=0, sequence_dim=0):
    """
    Group elements such that every group is a batch where all its 'input' tensors have the same length in a given dimension 'sequence_dim'.
    """
    max_batch_size = tf.constant(max_batch_size, tf.int32)
    min_batch_size = tf.constant(min_batch_size, tf.int32)
    sequence_dim = tf.constant(sequence_dim, tf.int32)
    def get_seq_len(x):
        return tf.cast(tf.shape(x["input"])[sequence_dim], tf.int64)
    def group_to_batch(key, group):
        return group.batch(max_batch_size)
    def has_min_batch_size(batch):
        return tf.shape(batch["input"])[0] >= min_batch_size
    return ds.apply(tf.data.experimental.group_by_window(
        get_seq_len,
        group_to_batch,
        window_size=max_batch_size))


def initialize(ds, labels, init_data):
    """
    Initialize a tf.data.Dataset instance for the pipeline.
    This should probably always be the first step.
    """
    if ds is not None:
        logger.warning("Step 'initialize' is being applied on an already initialized dataset, all state will be lost.")
    ds = None
    init_data_tensors = {key: tf.constant(list(meta), tf.string) for key, meta in init_data.items()}
    logger.info(
            "Initializing dataset from tensors with metadata keys:\n  %s",
            '\n  '.join(sorted(init_data_tensors.keys())))
    ds = tf.data.Dataset.from_tensor_slices(init_data_tensors)
    label2int, _ = tf_utils.make_label2onehot(tf.constant(labels, tf.string))
    logger.info(
            "Generated label2target lookup table from indexes of array:\n  %s",
            '\n  '.join("{:s} {:d}".format(l, label2int.lookup(tf.constant(l, tf.string))) for l in labels))
    append_labels_as_targets = lambda x: dict(x, target=label2int.lookup(x["label"]))
    return ds.map(append_labels_as_targets, num_parallel_calls=TF_AUTOTUNE)


def load_audio(ds):
    """
    Load signal from the 'path' key as WAV file for each element of ds.
    """
    logger.debug("Reading audio files from the path of each element and appending the read signals and their sample rates to each element.")
    def append_signals(x):
        signal, sample_rate = audio_features.read_wav(x["path"])
        return dict(x, signal=signal, sample_rate=sample_rate)
    return ds.map(append_signals, num_parallel_calls=TF_AUTOTUNE)


def lambda_fn(ds, fn):
    """
    For applying arbitrary logic on ds.
    """
    return fn(ds)


def reduce_stats(ds, statistic):
    """
    Reduce ds into a single statistic.
    This requires evaluating the full, preceding pipeline.
    """
    logger.debug("Iterating over whole dataset to compute statistic '%s'", statistic)
    if statistic == "num_elements":
        num_elements = ds.reduce(0, lambda c, x: c + 1)
        logger.info("Num elements: %d.", int(num_elements.numpy()))
    else:
        logger.error("Unknown statistic type '%s', cannot compute stats for dataset.", statistic)
    return ds


def show_all_elements(ds):
    """
    Iterate over ds printing every element.
    """
    for i, element in ds.enumerate(start=1).as_numpy_iterator():
        if not isinstance(element, dict):
            element = {i: v for i, v in enumerate(element)}
        logger.info("Element %d:\nshapes: %s\ncontents: %s.", i, repr(_element_shapes_dict(element)), repr(element))
    return ds


VALID_STEP_FUNCTIONS = {
    "append_predictions": append_predictions,
    "apply_filters": apply_filters,
    "apply_vad": apply_vad,
    "as_supervised": as_supervised,
    "cache": cache,
    "compute_webrtc_vad": compute_webrtc_vad,
    "consume": consume,
    "consume_to_tensorboard": consume_to_tensorboard,
    "create_signal_chunks": create_signal_chunks,
    "drop_empty": drop_empty,
    "drop_keys_in_set": drop_keys_in_set,
    "extract_features": extract_features,
    "filter_keys_in_set": filter_keys_in_set,
    "group_by_sequence_length": group_by_sequence_length,
    "initialize": initialize,
    "lambda": lambda_fn,
    "load_audio": load_audio,
    "reduce_stats": reduce_stats,
    "show_all_elements": show_all_elements,
}



# TODO random chunkers
#
# def make_random_frame_chunker_fn(len_config):
#     float_lengths = tf.linspace(
#         float(len_config["min"]),
#         float(len_config["max"]),
#         int(len_config["num_bins"]))
#     lengths = tf.unique(tf.cast(float_lengths, tf.int32))[0]
#     min_chunk_length = lengths[0]
#     tf.debugging.assert_greater(min_chunk_length, 1, message="Too short minimum chunk length")
#     min_overlap = tf.constant(float(len_config.get("min_overlap", 0)), tf.float32)
#     tf.debugging.assert_less(min_overlap, 1.0, message="Minimum overlap ratio of two adjacent random chunks must be less than 1.0")
#     @tf.function
#     def chunk_timedim_randomly(features):
#         num_total_frames = tf.shape(features)[0]
#         max_num_chunks = 1 + tf.math.maximum(0, num_total_frames - min_chunk_length)
#         rand_length_indexes = tf.random.uniform([max_num_chunks], 0, tf.size(lengths), dtype=tf.int32)
#         rand_chunk_lengths = tf.gather(lengths, rand_length_indexes)
#         rand_offset_ratios = tf.random.uniform([tf.size(rand_chunk_lengths)], 0.0, 1.0 - min_overlap, dtype=tf.float32)
#         offsets = tf.cast(rand_offset_ratios * tf.cast(rand_chunk_lengths, tf.float32), tf.int32)
#         offsets = tf.concat(([0], tf.math.maximum(1, offsets[:-1])), axis=0)
#         begin = tf.math.cumsum(offsets)
#         begin = tf.boolean_mask(begin, begin < num_total_frames)
#         end = begin + tf.boolean_mask(rand_chunk_lengths, begin < num_total_frames)
#         end = tf.math.minimum(num_total_frames, end)
#         # TODO gather is overkill here since we only want several slices
#         chunk_indices = tf.ragged.range(begin, end)
#         return tf.gather(features, chunk_indices)
#     return chunk_timedim_randomly

# TODO
# def get_random_chunk_loader(paths, meta, wav_config, verbosity=0):
#     raise NotImplementedError("todo, rewrite to return a function that supports tf.data.Dataset.interleave, like in get_chunk_loader")
#     chunk_config = wav_config["wav_to_random_chunks"]
#     sample_rate = wav_config["filter_sample_rate"]
#     lengths = tf.cast(
#         tf.linspace(
#             float(sample_rate * chunk_config["length"]["min"]),
#             float(sample_rate * chunk_config["length"]["max"]),
#             int(chunk_config["length"]["num_bins"]),
#         ),
#         tf.int32
#     )
#     def get_random_length():
#         rand_index = tf.random.uniform([], 0, tf.size(lengths), dtype=tf.int32)
#         rand_len = tf.gather(lengths, rand_index)
#         return rand_len
#     overlap_ratio = float(chunk_config["length"]["overlap_ratio"])
#     assert overlap_ratio < 1.0
#     min_chunk_length = int(sample_rate * chunk_config["min_chunk_length"])
#     assert min_chunk_length > 0, "invalid min chunk length"
#     def random_chunk_loader():
#         for p, *m in zip(paths, meta):
#             wav = audio_feat.read_wav(tf.constant(p, tf.string))
#             if "filter_sample_rate" in wav_config and wav.sample_rate != wav_config["filter_sample_rate"]:
#                 if verbosity:
#                     print("skipping file", p, ", it has a sample rate", wav.sample_rate, "but config has 'filter_sample_rate'", wav_config["filter_sample_rate"], file=sys.stderr)
#                 continue
#             begin = 0
#             rand_len = get_random_length()
#             chunk = wav.audio[begin:begin+rand_len]
#             while len(chunk) >= min_chunk_length:
#                 yield (audio_feat.Wav(chunk, wav.sample_rate), *m)
#                 begin += round((1.0 - overlap_ratio) * float(rand_len))
#                 rand_len = get_random_length()
#                 chunk = wav.audio[begin:begin+rand_len]
#     if verbosity:
#         tf_util.tf_print("Using random wav chunk loader, drawing lengths (in frames) from", lengths, "with", overlap_ratio, "overlap ratio and", min_chunk_length, "minimum chunk length")
#     return random_chunk_loader

# TODO loading of existing features in Kaldi format
#def parse_kaldi_features(utterance_list, features_path, utt2meta, expected_shape, feat_conf):
#    utt2label = {u: d["label"] for u, d in utt2meta.items()}
#    utt2feats = kaldiio.load_scp(features_path)
#    feat_conf = dict(feat_conf)
#    normalize_mean_axis = feat_conf.pop("normalize_mean_axis", None)
#    normalize_stddev_axis = feat_conf.pop("normalize_stddev_axis", None)
#    def assert_shape(shape):
#        shape_str = "{} vs {}".format(shape, expected_shape)
#        assert len(shape) == len(expected_shape), shape_str
#        assert all(x == y for x, y in zip(shape, expected_shape) if y is not None), shape_str
#    def datagen():
#        for utt in utterance_list:
#            if utt not in utt2feats:
#                print("warning: skipping utterance '{}' since it is not in the kaldi scp file".format(utt), file=sys.stderr)
#                continue
#            feats = utt2feats[utt]
#            #assert_shape(feats.shape)
#            normalized = tf.constant(feats, tf.float32)
#            if normalize_mean_axis is not None:
#                mean = tf.math.reduce_mean(feats, axis=normalize_mean_axis, keepdims=True)
#                normalized = feats - mean
#            if normalize_stddev_axis is not None:
#                stddev = tf.math.reduce_std(feats, axis=normalize_stddev_axis, keepdims=True)
#                normalized = tf.math.divide_no_nan(normalized, stddev)
#            yield normalized, (utt, utt2label[utt])
#    ds = tf.data.Dataset.from_generator(
#        datagen,
#        (tf.float32, tf.string),
#        (tf.TensorShape(expected_shape), tf.TensorShape([2])))
#    # Add empty signals (since we don't know the origin of these features)
#    empty_wav = audio_feat.Wav(tf.zeros([0]), 0)
#    add_empty_signal = lambda feats, meta: (feats, (meta, empty_wav))
#    return ds.map(add_empty_signal).batch(1)
