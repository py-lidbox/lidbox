import collections
import os
import random
import sys

import tensorflow as tf

import lidbox
if lidbox.TF_DEBUG:
    tf.autograph.set_verbosity(10, alsologtostdout=True)

from . import (
    audio_feat,
    parse_space_separated,
    system,
    tf_data,
    tf_serialization,
    yaml_pprint,
)


def tf_print(*args, **kwargs):
    if "summarize" not in kwargs:
        kwargs["summarize"] = -1
    if "output_stream" not in kwargs:
        kwargs["output_stream"] = sys.stdout
    return tf.print(*args, **kwargs)


@tf.function
def count_dim_sizes(ds, ds_element_index=0, ndims=1):
    """
    Given a dataset 'ds' of 'ndims' dimensional tensors at element index 'ds_element_index', accumulate the shape counts of all tensors.
    >>> batch, x, y = 10, 3, 4
    >>> data = tf.random.normal((batch, x, y))
    >>> meta = tf.random.normal((batch, 1))
    >>> ds = tf.data.Dataset.from_tensor_slices((data, meta))
    >>> data_sizes = count_dim_sizes(ds, 0, 2)
    >>> x_size = tf.squeeze(data_sizes[0], 0)
    >>> tf.math.reduce_all(x_size == [batch, x]).numpy()
    True
    >>> y_size = tf.squeeze(data_sizes[1], 0)
    >>> tf.math.reduce_all(y_size == [batch, y]).numpy()
    True
    >>> meta_size = tf.squeeze(count_dim_sizes(ds, 1, 1), [0, 1])
    >>> tf.math.reduce_all(meta_size == [batch, 1]).numpy()
    True
    """
    tf.debugging.assert_greater(ndims, 0)
    get_shape_at_index = lambda *t: tf.shape(t[ds_element_index])
    shapes_ds = ds.map(get_shape_at_index)
    ones = tf.ones(ndims, dtype=tf.int32)
    shape_indices = tf.range(ndims, dtype=tf.int32)
    max_sizes = shapes_ds.reduce(
        tf.zeros(ndims, dtype=tf.int32),
        lambda acc, shape: tf.math.maximum(acc, shape))
    max_max_size = tf.reduce_max(max_sizes)
    @tf.function
    def accumulate_dim_size_counts(counter, shape):
        enumerated_shape = tf.stack((shape_indices, shape), axis=1)
        return tf.tensor_scatter_nd_add(counter, enumerated_shape, ones)
    size_counts = shapes_ds.reduce(
        tf.zeros((ndims, max_max_size + 1), dtype=tf.int32),
        accumulate_dim_size_counts)
    sorted_size_indices = tf.argsort(size_counts, direction="DESCENDING")
    sorted_size_counts = tf.gather(size_counts, sorted_size_indices, batch_dims=1)
    is_nonzero = sorted_size_counts > 0
    return tf.ragged.stack(
        (tf.ragged.boolean_mask(sorted_size_counts, is_nonzero),
         tf.ragged.boolean_mask(sorted_size_indices, is_nonzero)),
        axis=2)


def make_label2onehot(labels):
    """
    >>> labels = tf.constant(["one", "two", "three"], tf.string)
    >>> label2int, OH = make_label2onehot(labels)
    >>> for i in range(3):
    ...     (label2int.lookup(labels[i]).numpy(), tf.math.argmax(OH[i]).numpy())
    (0, 0)
    (1, 1)
    (2, 2)
    """
    labels_enum = tf.range(len(labels))
    # Label to int or one past last one if not found
    # TODO slice index out of bounds is probably not a very informative error message
    label2int = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(labels),
            tf.constant(labels_enum)
        ),
        tf.constant(len(labels), dtype=tf.int32)
    )
    OH = tf.one_hot(labels_enum, len(labels))
    return label2int, OH


def extract_features(datasets, config, datagroup_key, verbosity=1, force_shuffle_utt2path=False, file_limit=None, **kwargs):
    utt2path = collections.OrderedDict()
    utt2meta = collections.OrderedDict()
    if verbosity > 1:
        print("Extracting features from datagroup '{}'".format(datagroup_key))
        if verbosity > 2:
            yaml_pprint(config)
    num_utts_dropped = collections.Counter()
    for ds_config in datasets:
        if verbosity > 1:
            print("Dataset '{}'".format(ds_config["key"]))
        datagroup = ds_config["datagroups"][datagroup_key]
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        if verbosity:
            print("Reading labels for utterances from utt2label file '{}'".format(utt2label_path))
        if verbosity > 1:
            print("Expected labels (utterances with other labels will be ignored):")
            for l in ds_config["labels"]:
                print("  {}".format(l))
        enabled_labels = set(ds_config["labels"])
        skipped_utterances = set()
        for utt, label, *rest in parse_space_separated(utt2label_path):
            if label not in enabled_labels:
                skipped_utterances.add(utt)
                continue
            assert utt not in utt2meta, "duplicate utterance id found when parsing labels: '{}'".format(utt)
            utt2meta[utt] = {"label": label, "dataset": ds_config["key"], "duration_sec": -1.0}
        utt2dur_path = os.path.join(datagroup["path"], datagroup.get("utt2dur", "utt2dur"))
        if os.path.exists(utt2dur_path):
            if verbosity:
                print("Reading durations from utt2dur file '{}'".format(utt2dur_path))
            for utt, duration, *rest in parse_space_separated(utt2dur_path):
                if utt in skipped_utterances:
                    continue
                assert utt in utt2meta, "utterance id without label found when parsing durations: '{}'".format(utt)
                utt2meta[utt]["duration_sec"] = float(duration)
        else:
            if verbosity:
                print("Skipping signal duration parse since utt2dur file '{}' does not exist".format(utt2dur_path))
        if verbosity:
            print("Reading paths of wav files from utt2path file '{}'".format(utt2path_path))
        for utt, path, *rest in parse_space_separated(utt2path_path):
            if utt in skipped_utterances:
                continue
            assert utt not in utt2path, "duplicate utterance id found when parsing paths: '{}'".format(utt)
            utt2path[utt] = path
    if verbosity > 1:
        print("Total amount of non-empty lines read from utt2path {}, and utt2meta {}".format(len(utt2path), len(utt2meta)))
        if skipped_utterances:
            print("Utterances skipped due to unexpected labels: {}".format(len(skipped_utterances)))
    # All utterance ids must be present in both files
    assert set(utt2path) == set(utt2meta), "Mismatching sets of utterances in utt2path and utt2meta, the utterance ids must be exactly the same"
    utterance_list = list(utt2path.keys())
    if force_shuffle_utt2path or datagroup.get("shuffle_utt2path", False):
        if verbosity > 1:
            print("Shuffling utterance ids, all wavpaths in the utt2path list will be processed in random order.")
        random.shuffle(utterance_list)
    else:
        if verbosity > 1:
            print("Not shuffling utterance ids, all wavs will be processed in order of the utt2path list.")
    if file_limit is not None:
        if verbosity > 1:
            print("--file-limit set at {0}, using at most {0} utterances from the utterance id list, starting at the beginning of utt2path".format(file_limit))
        utterance_list = utterance_list[:file_limit]
        if verbosity > 3:
            print("Using utterance ids:")
            yaml_pprint(utterance_list)
    paths = []
    paths_meta = []
    for utt in utterance_list:
        paths.append(utt2path[utt])
        meta = utt2meta[utt]
        paths_meta.append((utt, meta["label"], meta["dataset"], meta["duration_sec"]))
    if verbosity:
        print("Starting feature extraction for datagroup '{}' from {} files".format(datagroup_key, len(paths)))
        if verbosity > 3:
            print("Debug verbosity specified, dumping all utterances:")
            print("{:>30s} {:>10s} {:>20s}".format("utt", "label", "dataset"))
            for path, (utt, label, dataset, *rest) in zip(paths, paths_meta):
                print("{:>30s} {:>10s} {:>20s}".format(utt, label, dataset))
    if config["type"] == "sparsespeech":
        seg2utt_path = os.path.join(datagroup["path"], "segmented", datagroup.get("seg2utt", "seg2utt"))
        if verbosity:
            print("Parsing SparseSpeech features")
            print("Reading utterance segmentation data from seg2utt file '{}'".format(seg2utt_path))
        seg2utt = collections.OrderedDict(
            row[:2] for row in parse_space_separated(seg2utt_path))
        enc_path = config["sparsespeech_paths"]["output"][datagroup_key]
        feat_path = config["sparsespeech_paths"]["input"][datagroup_key]
        if verbosity:
            print("SparseSpeech input: '{}' and encoding: '{}'".format(feat_path, enc_path))
        feat = tf_data.parse_sparsespeech_features(config, enc_path, feat_path, seg2utt, utt2label)
    elif config["type"] == "kaldi":
        feat_conf = dict(config["datagroups"][datagroup_key])
        kaldi_feats_scp = feat_conf.pop("features_path")
        expected_shape = feat_conf.pop("shape")
        if verbosity:
            print("Parsing Kaldi features from '{}' with expected shape {}".format(kaldi_feats_scp, expected_shape))
        feat = tf_data.parse_kaldi_features(utterance_list, kaldi_feats_scp, utt2label, expected_shape, feat_conf)
    else:
        feat = tf_data.extract_features_from_paths(
            config,
            paths,
            paths_meta,
            datagroup_key,
            verbosity=verbosity)
    return feat


def extract_features_with_cache(config, experiment_config, datagroup_key, cache_dir, **kwargs):
    verbosity = kwargs["verbosity"]
    feat_config = experiment_config["features"]
    conf_checksum = kwargs["conf_checksum"]
    conf_json = kwargs["conf_json"]
    extractor_ds = extract_features(
        experiment_config["datasets"],
        feat_config,
        datagroup_key,
        **kwargs)
    features_cache_dir = os.path.join(
        cache_dir,
        "features",
        datagroup_key,
        feat_config["type"],
        conf_checksum)
    num_shards = config["features_cache"]["num_shards"]
    cache_exists = os.path.exists(features_cache_dir + ".md5sum-input")
    if cache_exists:
        if verbosity:
            print("Loading features from existing cache: '{}'".format(features_cache_dir))
    else:
        if verbosity:
            print("Writing features into new cache: '{}'".format(features_cache_dir)
                    + '' if num_shards == 1 else " using {} shards".format(num_shards))
        os.makedirs(features_cache_dir)
        with open(features_cache_dir + ".md5sum-input", "w") as f:
            print(conf_json, file=f, end='')
    if config["features_cache"].get("include_signals", True):
        if verbosity:
            print("Original signals will be included in the features cache")
    else:
        if verbosity:
            print("Original signals will be dropped before saving features into cache")
        extractor_ds = extractor_ds.map(lambda features, meta: (features, meta[:-1]))
    if num_shards == 1:
        return extractor_ds.cache(filename=os.path.join(features_cache_dir, "all_features"))
    shards = [extractor_ds.shard(num_shards, i) for i in range(num_shards)]
    cached_ds = shards[0]
    for shard in shards[1:]:
        cached_ds = cached_ds.concatenate(shard)
    return cached_ds


def extract_features_with_tfrecords(config, experiment_config, datagroup_key, cache_dir, **kwargs):
    verbosity = kwargs["verbosity"]
    feat_config = experiment_config["features"]
    conf_checksum = kwargs["conf_checksum"]
    conf_json = kwargs["conf_json"]
    features_cache_dir = os.path.join(
        cache_dir,
        "features",
        datagroup_key,
        feat_config["type"],
        conf_checksum)
    cache_exists = os.path.exists(features_cache_dir + ".md5sum-input")
    if not cache_exists:
        if verbosity:
            print("Writing features into new cache: '{}'".format(features_cache_dir))
        os.makedirs(features_cache_dir, exist_ok=True)
        with open(features_cache_dir + ".md5sum-input", "w") as f:
            print(conf_json, file=f, end='')
        tmp_ds = extract_features(
            experiment_config["datasets"],
            feat_config,
            datagroup_key,
            **kwargs)
        tfrecord_paths = tf_serialization.write_features(tmp_ds, features_cache_dir)
        for i, path in tfrecord_paths.enumerate().as_numpy_iterator():
            if verbosity:
                print("Wrote TFRecord {} to '{}'".format(i, path))
    else:
        if verbosity:
            print("Loading features from existing cache: '{}'".format(features_cache_dir))
    #TODO load from metadata file
    wav_length = 15680
    return tf_serialization.load_features(
            features_cache_dir,
            experiment_config["experiment"]["input_shape"][1],
            wav_length)
