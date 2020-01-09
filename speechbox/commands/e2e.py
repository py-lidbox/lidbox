import collections
import hashlib
import itertools
import importlib
import json
import os
import random
import sys
import time

import kaldiio
import numpy as np
import tensorflow as tf

from speechbox import yaml_pprint
from speechbox.commands.base import Command, State, StatefulCommand
from speechbox.metrics import AverageDetectionCost, AverageEqualErrorRate, AveragePrecision, AverageRecall
import speechbox.models as models
import speechbox.tf_data as tf_data
import speechbox.system as system


class E2E(Command):
    """TensorFlow pipeline for wavfile preprocessing, feature extraction and sound classification model training"""
    pass


def parse_space_separated(path):
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                yield l.split()

def make_label2onehot(labels):
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

def config_checksum(config, datagroup_key):
    md5input = {
        "features": config["features"],
        "wav_config": config["dataset"]["datagroups"][datagroup_key],
    }
    json_str = json.dumps(md5input, ensure_ascii=False, sort_keys=True) + '\n'
    return json_str, hashlib.md5(json_str.encode("utf-8")).hexdigest()

def count_dim_sizes(ds, ds_element_index, ndims):
    tf.debugging.assert_greater(ndims, 0)
    shapes_ds = ds.map(lambda *t: tf.shape(t[ds_element_index])).cache()
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


class E2EBase(StatefulCommand):
    requires_state = State.none

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("options")
        optional.add_argument("--file-limit",
            type=int,
            help="Extract only up to this many files from the wavpath list (e.g. for debugging).")
        return parser

    def get_checkpoint_dir(self):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        return os.path.join(model_cache_dir, "checkpoints")

    def create_model(self, config, skip_training=False):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        tensorboard_log_dir = os.path.join(model_cache_dir, "tensorboard", "logs")
        now_str = str(int(time.time()))
        tensorboard_dir = os.path.join(tensorboard_log_dir, now_str)
        default_tensorboard_config = {
            "log_dir": tensorboard_dir,
            "profile_batch": 0,
            "histogram_freq": 1,
        }
        tensorboard_config = dict(default_tensorboard_config, **config.get("tensorboard", {}))
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoint_format = "epoch{epoch:06d}.hdf5"
        if "checkpoints" in config and "format" in config["checkpoints"]:
            checkpoint_format = config["checkpoints"].pop("format")
        default_checkpoints_config = {
            "filepath": os.path.join(checkpoint_dir, checkpoint_format),
        }
        checkpoints_config = dict(default_checkpoints_config, **config.get("checkpoints", {}))
        callbacks_kwargs = {
            "checkpoints": checkpoints_config,
            "early_stopping": config.get("early_stopping"),
            "tensorboard": tensorboard_config,
            "other_callbacks": config.get("other_callbacks", []),
        }
        if not skip_training:
            self.make_named_dir(tensorboard_dir, "tensorboard")
            self.make_named_dir(checkpoint_dir, "checkpoints")
        if self.args.verbosity > 1:
            print("KerasWrapper callback parameters will be set to:")
            yaml_pprint(callbacks_kwargs)
            print()
        return models.KerasWrapper(self.model_id, config["model_definition"], **callbacks_kwargs)

    def extract_features(self, config, datagroup_key, trim_audio, debug_squeeze_last_dim):
        args = self.args
        datagroup = self.experiment_config["dataset"]["datagroups"][datagroup_key]
        if args.verbosity > 2:
            print("Extracting features from datagroup '{}' with config".format(datagroup_key))
            yaml_pprint(config)
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        if args.verbosity:
            print("Reading paths of wav files from utt2path file '{}'".format(utt2path_path))
        utt2path = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2path_path)
        )
        if args.verbosity:
            print("Reading labels for utterances from utt2label file '{}'".format(utt2label_path))
        utt2label = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2label_path)
        )
        if args.verbosity > 1:
            print("Non-empty lines read from utt2path {}, and utt2label {}".format(len(utt2path), len(utt2label)))
        # All utterance ids must be present in both files
        assert set(utt2path) == set(utt2label), "utt2path and utt2label must have exactly matching sets of utterance ids"
        utterance_list = list(utt2path.keys())
        if datagroup.get("shuffle_utt2path", False):
            if args.verbosity > 1:
                print("Shuffling utterance ids, all wavpaths in the utt2path list will be processed in random order.")
            random.shuffle(utterance_list)
        else:
            if args.verbosity > 1:
                print("Not shuffling utterance ids, all wavs will be processed in order of the utt2path list.")
        if args.file_limit:
            if args.verbosity > 1:
                print("--file-limit set at {0}, using at most {0} utterances from the utterance id list, starting at the beginning of utt2path".format(args.file_limit))
            utterance_list = utterance_list[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                yaml_pprint(utterance_list)
        labels_set = set(self.experiment_config["dataset"]["labels"])
        num_dropped = collections.Counter()
        paths = []
        paths_meta = []
        for utt in utterance_list:
            label = utt2label[utt]
            if label not in labels_set:
                num_dropped[label] += 1
                continue
            paths.append(utt2path[utt])
            paths_meta.append((utt, label))
        if args.verbosity:
            print("Starting feature extraction for datagroup '{}' from {} files".format(datagroup_key, len(paths)))
            if num_dropped:
                print("Amount of files ignored since they had a label that was not in the labels list dataset.labels: {}".format(num_dropped.most_common(None)))
            if args.verbosity > 3:
                print("Using paths:")
                for path, (utt, label) in zip(paths, paths_meta):
                    print(utt, path, label)
        if config["type"] == "sparsespeech":
            seg2utt_path = os.path.join(datagroup["path"], "segmented", datagroup.get("seg2utt", "seg2utt"))
            if args.verbosity:
                print("Parsing SparseSpeech features")
                print("Reading utterance segmentation data from seg2utt file '{}'".format(seg2utt_path))
            seg2utt = collections.OrderedDict(
                row[:2] for row in parse_space_separated(seg2utt_path)
            )
            enc_path = config["sparsespeech_paths"]["output"][datagroup_key]
            feat_path = config["sparsespeech_paths"]["input"][datagroup_key]
            if args.verbosity:
                print("SparseSpeech input: '{}' and encoding: '{}'".format(feat_path, enc_path))
            feat = tf_data.parse_sparsespeech_features(config, enc_path, feat_path, seg2utt, utt2label)
        elif config["type"] == "kaldi":
            kaldi_feats_scp = config["datagroups"][datagroup_key]["features_path"]
            expected_shape = config["datagroups"][datagroup_key]["shape"]
            if args.verbosity:
                print("Parsing Kaldi features from '{}' with expected shape {}".format(kaldi_feats_scp, expected_shape))
            feat = tf_data.parse_kaldi_features(utterance_list, kaldi_feats_scp, utt2label, expected_shape)
        else:
            feat = tf_data.extract_features_from_paths(
                config,
                self.experiment_config["dataset"]["datagroups"][datagroup_key],
                paths,
                paths_meta,
                verbosity=args.verbosity,
                trim_audio=trim_audio,
                debug_squeeze_last_dim=debug_squeeze_last_dim,
            )
        return feat

class Train(E2EBase):

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("options")
        optional.add_argument("--skip-training",
            action="store_true",
            default=False)
        optional.add_argument("--debug-dataset",
            action="store_true",
            default=False)
        optional.add_argument("--exhaust-dataset-iterator",
            action="store_true",
            default=False,
            help="Explictly iterate once over the feature extractor tf.data.Dataset object in order to evaluate the feature extraction pipeline and fill the feature cache on disk. Using this with --skip-training allows you to extract features on multiple CPUs without needing a GPU.")
        return parser

    def train(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for training")
        training_config = self.experiment_config["experiment"]
        feat_config = self.experiment_config["features"]
        if args.verbosity > 1:
            print("Using model parameters:")
            yaml_pprint(training_config)
            print()
        if args.verbosity > 1:
            print("Using feature extraction parameters:")
            yaml_pprint(feat_config)
            print()
        if feat_config["type"] == "xvector-embedding":
            xvec_config = system.load_yaml(feat_config["xvector_experiment_config"])
            feat_config = xvec_config.pop("features")
        else:
            xvec_config = {}
        labels = self.experiment_config["dataset"]["labels"]
        label2int, OH = make_label2onehot(labels)
        label2onehot = lambda label: OH[label2int.lookup(label)]
        if args.verbosity > 2:
            print("Generating onehot encoding from labels:", ', '.join(labels))
            print("Generated onehot encoding as tensors:")
            for l in labels:
                l = tf.constant(l, dtype=tf.string)
                tf_data.tf_print(l, "\t", label2onehot(l))
        self.model_id = training_config["name"]
        model = self.create_model(dict(training_config), args.skip_training)
        dataset = {}
        for ds in ("train", "validation"):
            if args.verbosity > 2:
                print("Dataset config for '{}'".format(ds))
                yaml_pprint(training_config[ds])
            ds_config = dict(training_config, **training_config[ds])
            del ds_config["train"], ds_config["validation"]
            summary_kwargs = dict(ds_config.get("dataset_logger", {}))
            debug_squeeze_last_dim = ds_config["input_shape"][-1] == 1
            datagroup_key = ds_config.pop("datagroup")
            extractor_ds = self.extract_features(
                feat_config,
                datagroup_key,
                summary_kwargs.pop("trim_audio", False),
                debug_squeeze_last_dim,
            )
            if xvec_config:
                if args.verbosity:
                    print("Features extracted, now feeding them to the xvector encoder")
                xvec_model = models.KerasWrapper(
                    xvec_config["experiment"]["name"],
                    xvec_config["experiment"]["model_definition"],
                )
                xvec_model.prepare(
                    xvec_config["dataset"]["labels"],
                    xvec_config["experiment"],
                )
                xvec_weights = os.path.join(
                    xvec_config["cache"],
                    xvec_config["experiment"]["name"],
                    "checkpoints",
                    xvec_config["prediction"]["best_checkpoint"],
                )
                xvec_model.load_weights(xvec_weights)
                xvector_extractor = tf.keras.Sequential(
                    [xvec_model.model.get_layer(l) for l in importlib.import_module("speechbox.models.xvector").xvector_layer_names]
                )
                if args.verbosity > 1:
                    print("Xvector extractor is:\n", xvector_extractor)
                embed_xvec = lambda feats, *meta: (tf.expand_dims(xvector_extractor(feats), -2), *meta)
                extractor_ds = extractor_ds.batch(1).map(embed_xvec).unbatch()
            if ds_config.get("cache_features_in_tmp", False):
                features_cache_dir = "/tmp"
            else:
                features_cache_dir = os.path.join(self.cache_dir, "features")
            conf_json, conf_checksum = config_checksum(self.experiment_config, datagroup_key)
            features_cache_path = os.path.join(
                features_cache_dir,
                self.experiment_config["dataset"]["key"],
                ds,
                feat_config["type"],
                conf_checksum,
            )
            self.make_named_dir(os.path.dirname(features_cache_path), "features cache")
            if not os.path.exists(features_cache_path + ".md5sum-input"):
                with open(features_cache_path + ".md5sum-input", "w") as f:
                    print(conf_json, file=f, end='')
                if args.verbosity:
                    print("Writing features into new cache: '{}'".format(features_cache_path))
            else:
                if args.verbosity:
                    print("Loading features from existing cache: '{}'".format(features_cache_path))
            extractor_ds = extractor_ds.cache(filename=features_cache_path)
            if args.exhaust_dataset_iterator:
                if args.verbosity:
                    print("--exhaust-dataset-iterator given, now iterating once over the dataset iterator to fill the features cache.")
                # This forces the extractor_ds pipeline to be evaluated, and the features being serialized into the cache
                i = 0
                for i, (feats, meta, *rest) in enumerate(extractor_ds):
                    if args.verbosity > 1 and i % 2000 == 0:
                        print(i, "samples done")
                    if args.verbosity > 3:
                        tf_data.tf_print("sample:", i, "features shape:", tf.shape(feats), "metadata:", meta)
                if args.verbosity > 1:
                    print("all", i, "samples done")
            if args.verbosity > 2:
                print("Preparing dataset iterator for training")
            dataset[ds] = tf_data.prepare_dataset_for_training(
                extractor_ds,
                ds_config,
                feat_config,
                label2onehot,
                conf_checksum=conf_checksum,
                verbosity=args.verbosity,
            )
            if args.debug_dataset:
                if args.verbosity:
                    print("--debug-dataset given, iterating over the dataset to gather stats")
                if args.verbosity > 1:
                    print("Counting all unique dim sizes of elements at index 0 in dataset")
                for axis, size_counts in enumerate(count_dim_sizes(dataset[ds], 0, len(ds_config["input_shape"]) + 1)):
                    print("axis {}\n[count size]:".format(axis))
                    tf_data.tf_print(size_counts, summarize=10)
                if summary_kwargs:
                    logdir = os.path.join(os.path.dirname(model.tensorboard.log_dir), "dataset", ds)
                    if os.path.isdir(logdir):
                        if args.verbosity:
                            print("summary_kwargs available, but '{}' already exists, not iterating over dataset again".format(logdir))
                    else:
                        if args.verbosity:
                            print("Datagroup '{}' has a dataset logger defined. We will iterate over {} batches of samples from the dataset to create TensorBoard summaries of the input data into '{}'.".format(ds, summary_kwargs.get("num_batches", "'all'"), logdir))
                        self.make_named_dir(logdir)
                        writer = tf.summary.create_file_writer(logdir)
                        summary_kwargs["debug_squeeze_last_dim"] = debug_squeeze_last_dim
                        with writer.as_default():
                            logged_dataset = tf_data.attach_dataset_logger(dataset[ds], feat_config["type"], **summary_kwargs)
                            if args.verbosity:
                                print("Dataset logger attached to '{0}' dataset iterator, now exhausting the '{0}' dataset logger iterator once to write TensorBoard summaries of model input data".format(ds))
                            i = 0
                            for i, elem in enumerate(logged_dataset):
                                if args.verbosity > 1 and i % (2000//ds_config.get("batch_size", 1)) == 0:
                                    print(i, "batches done")
                            if args.verbosity > 1:
                                print(i, "batches done")
                            del logged_dataset
        if args.verbosity > 1:
            print("Preparing model")
        model.prepare(labels, training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoints = [c.name for c in os.scandir(checkpoint_dir) if c.is_file()] if os.path.isdir(checkpoint_dir) else []
        if checkpoints:
            if "checkpoints" in training_config:
                monitor_value = training_config["checkpoints"]["monitor"]
                monitor_mode = training_config["checkpoints"].get("mode")
            else:
                monitor_value = "epoch"
                monitor_mode = None
            checkpoint_path = os.path.join(checkpoint_dir, models.get_best_checkpoint(checkpoints, key=monitor_value, mode=monitor_mode))
            if args.verbosity:
                print("Loading model weights from checkpoint file '{}' according to monitor value '{}'".format(checkpoint_path, monitor_value))
            model.load_weights(checkpoint_path)
        if args.verbosity:
            print("\nStarting training with:")
            print(str(model))
            print()
        if args.skip_training:
            print("--skip-training given, will not call model.fit")
            return
        history = model.fit(dataset["train"], dataset["validation"], training_config)
        if args.verbosity:
            print("\nTraining finished after {} epochs at epoch {}".format(len(history.epoch), history.epoch[-1] + 1))
            print("metric:\tmin (epoch),\tmax (epoch):")
            for name, epoch_vals in history.history.items():
                vals = np.array(epoch_vals)
                print("{}:\t{:.6f} ({:d}),\t{:.6f} ({:d})".format(
                    name,
                    vals.min(),
                    vals.argmin() + 1,
                    vals.max(),
                    vals.argmax() + 1
                ))

    def run(self):
        super().run()
        return self.train()


class Predict(E2EBase):
    """
    Use a trained model to produce likelihoods for all target languages from all utterances in the test set.
    Writes all predictions as scores and information about the target and non-target languages into the cache dir.
    """

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("predict options")
        optional.add_argument("--score-precision", type=int, default=6)
        optional.add_argument("--score-separator", type=str, default=' ')
        optional.add_argument("--trials", type=str)
        optional.add_argument("--scores", type=str)
        optional.add_argument("--checkpoint",
            type=str,
            help="Specify which Keras checkpoint to load model weights from, instead of using the most recent one.")
        return parser

    def predict(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for prediction")
        self.model_id = self.experiment_config["experiment"]["name"]
        if not args.trials:
            args.trials = os.path.join(self.cache_dir, self.model_id, "predictions", "trials")
        if not args.scores:
            args.scores = os.path.join(self.cache_dir, self.model_id, "predictions", "scores")
        self.make_named_dir(os.path.dirname(args.trials))
        self.make_named_dir(os.path.dirname(args.scores))
        training_config = self.experiment_config["experiment"]
        feat_config = self.experiment_config["features"]
        if args.verbosity > 1:
            print("Using model parameters:")
            yaml_pprint(training_config)
            print()
        if args.verbosity > 1:
            print("Using feature extraction parameters:")
            yaml_pprint(feat_config)
            print()
        model = self.create_model(dict(training_config), skip_training=True)
        if args.verbosity > 1:
            print("Preparing model")
        labels = self.experiment_config["dataset"]["labels"]
        model.prepare(labels, training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        if args.checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
        elif "best_checkpoint" in self.experiment_config.get("prediction", {}):
            checkpoint_path = os.path.join(checkpoint_dir, self.experiment_config["prediction"]["best_checkpoint"])
        else:
            checkpoints = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
            if not checkpoints:
                print("Error: Cannot evaluate with a model that has no checkpoints, i.e. is not trained.")
                return 1
            if "checkpoints" in training_config:
                monitor_value = training_config["checkpoints"]["monitor"]
                monitor_mode = training_config["checkpoints"].get("mode")
            else:
                monitor_value = "epoch"
                monitor_mode = None
            checkpoint_path = os.path.join(checkpoint_dir, models.get_best_checkpoint(checkpoints, key=monitor_value, mode=monitor_mode))
        if args.verbosity:
            print("Loading model weights from checkpoint file '{}'".format(checkpoint_path))
        model.load_weights(checkpoint_path)
        if args.verbosity:
            print("\nEvaluating testset with model:")
            print(str(model))
            print()
        ds = "test"
        if args.verbosity > 2:
            print("Dataset config for '{}'".format(ds))
            yaml_pprint(training_config[ds])
        ds_config = dict(training_config, **training_config[ds])
        del ds_config["train"], ds_config["validation"]
        if args.verbosity and "dataset_logger" in ds_config:
            print("Warning: dataset_logger in the test datagroup has no effect.")
        datagroup_key = ds_config.pop("datagroup")
        datagroup = self.experiment_config["dataset"]["datagroups"][datagroup_key]
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        utt2path = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2path_path)
        )
        utt2label = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2label_path)
        )
        utterance_list = list(utt2path.keys())
        if args.file_limit:
            utterance_list = utterance_list[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                yaml_pprint(utterance_list)
        int2label = self.experiment_config["dataset"]["labels"]
        label2int, OH = make_label2onehot(int2label)
        label2onehot = lambda label: OH[label2int.lookup(label)]
        labels_set = set(int2label)
        paths = []
        paths_meta = []
        for utt in utterance_list:
            label = utt2label[utt]
            if label not in labels_set:
                continue
            paths.append(utt2path[utt])
            paths_meta.append((utt, label))
        if args.verbosity:
            print("Extracting test set features for prediction")
        features = self.extract_features(
            feat_config,
            "test",
            trim_audio=False,
            debug_squeeze_last_dim=(ds_config["input_shape"][-1] == 1),
        )
        conf_json, conf_checksum = config_checksum(self.experiment_config, datagroup_key)
        features = tf_data.prepare_dataset_for_training(
            features,
            ds_config,
            feat_config,
            label2onehot,
            verbosity=args.verbosity,
            conf_checksum=conf_checksum,
        )
        # drop meta wavs required only for vad
        features = features.map(lambda *t: t[:3])
        if ds_config.get("cache_features_in_tmp", False):
            features_cache_dir = "/tmp"
        else:
            features_cache_dir = os.path.join(self.cache_dir, "features")
        features_cache_path = os.path.join(
            features_cache_dir,
            self.experiment_config["dataset"]["key"],
            ds,
            feat_config["type"],
            conf_checksum,
        )
        if args.verbosity > 1:
            print("Caching feature extractor dataset to '{}'".format(features_cache_path))
        self.make_named_dir(os.path.dirname(features_cache_path), "features cache")
        with open(features_cache_path + ".md5sum-input", "w") as f:
            print(conf_json, file=f, end='')
        features = features.cache(filename=features_cache_path)
        if args.verbosity:
            print("Starting feature extraction")
        # Gather utterance ids, this also causes the extraction pipeline to be evaluated
        utterance_ids = [uttid.decode("utf-8") for _, _, uttids in features for uttid in uttids.numpy()]
        if args.verbosity:
            print("Features extracted, writing target and non-target language information for each utterance to '{}'.".format(args.trials))
        with open(args.trials, "w") as trials_f:
            for utt, target in utt2label.items():
                for lang in int2label:
                    print(lang, utt, "target" if target == lang else "nontarget", file=trials_f)
        if args.verbosity:
            print("Starting prediction")
        predictions = model.predict(features)
        if args.verbosity > 1:
            print("Done predicting, model returned predictions of shape {}. Writing them to '{}'.".format(predictions.shape, args.scores))
        num_predictions = 0
        with open(args.scores, "w") as scores_f:
            print(*int2label, file=scores_f)
            for utt, pred in zip(utterance_ids, predictions):
                pred_scores = [np.format_float_positional(x, precision=args.score_precision) for x in pred]
                print(utt, *pred_scores, sep=args.score_separator, file=scores_f)
                num_predictions += 1
        if args.verbosity:
            print("Wrote {} prediction scores to '{}'.".format(num_predictions, args.scores))

    def run(self):
        super().run()
        return self.predict()


class Evaluate(E2EBase):
    """Evaluate predicted scores by average detection cost (C_avg)."""

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("evaluate options")
        optional.add_argument("--trials", type=str)
        optional.add_argument("--scores", type=str)
        optional.add_argument("--threshold-bins", type=int, default=40)
        optional.add_argument("--convert-scores", choices=("softmax", "exp", "none"), default=None)
        return parser

    #TODO tf is very slow in the for loop, maybe numpy would be sufficient
    def evaluate(self):
        args = self.args
        self.model_id = self.experiment_config["experiment"]["name"]
        if not args.trials:
            args.trials = os.path.join(self.cache_dir, self.model_id, "predictions", "trials")
        if not args.scores:
            args.scores = os.path.join(self.cache_dir, self.model_id, "predictions", "scores")
        if args.verbosity > 1:
            print("Evaluating minimum average detection cost using trials '{}' and scores '{}'".format(args.trials, args.scores))
        score_lines = list(parse_space_separated(args.scores))
        langs = score_lines[0]
        lang2int = {l: i for i, l in enumerate(langs)}
        utt2scores = {utt: tf.constant([float(s) for s in scores], dtype=tf.float32) for utt, *scores in score_lines[1:]}
        if args.verbosity > 1:
            print("Parsed scores for {} utterances".format(len(utt2scores)))
        if args.convert_scores == "softmax":
            print("Applying softmax on logit scores")
            utt2scores = {utt: tf.keras.activations.softmax(tf.expand_dims(scores, 0))[0] for utt, scores in utt2scores.items()}
        elif args.convert_scores == "exp":
            print("Applying exp on log likelihood scores")
            utt2scores = {utt: tf.math.exp(scores) for utt, scores in utt2scores.items()}
        if args.verbosity > 2:
            print("Asserting all scores sum to 1")
            tolerance = 1e-3
            for utt, scores in utt2scores.items():
                one = tf.constant(1.0, dtype=tf.float32)
                tf.debugging.assert_near(
                    tf.reduce_sum(scores),
                    one,
                    rtol=tolerance,
                    atol=tolerance,
                    message="failed to convert log likelihoods to probabilities, the probabilities of predictions for utterance '{}' does not sum to 1".format(utt))
        if args.verbosity > 1:
            print("Generating {} threshold bins".format(args.threshold_bins))
        assert args.threshold_bins > 0
        from_logits = False
        max_score = tf.constant(-float("inf"), dtype=tf.float32)
        min_score = tf.constant(float("inf"), dtype=tf.float32)
        for utt, scores in utt2scores.items():
            max_score = tf.math.maximum(max_score, tf.math.reduce_max(scores))
            min_score = tf.math.minimum(min_score, tf.math.reduce_min(scores))
        if args.verbosity > 2:
            tf_data.tf_print("Max score", max_score, "min score", min_score)
        thresholds = tf.linspace(min_score, max_score, args.threshold_bins)
        if args.verbosity > 2:
            print("Score thresholds for language detection decisions:")
            tf_data.tf_print(thresholds, summarize=5)
        # First do C_avg to get the best threshold
        cavg = AverageDetectionCost(langs, theta_det=list(thresholds.numpy()))
        if args.verbosity > 1:
            print("Sorting trials")
        trials_by_utt = sorted(parse_space_separated(args.trials), key=lambda t: t[1])
        trials_by_utt = [
            (utt, tf.constant([float(t == "target") for _, _, t in sorted(group, key=lambda t: lang2int[t[0]])], dtype=tf.float32))
            for utt, group in itertools.groupby(trials_by_utt, key=lambda t: t[1])
        ]
        if args.verbosity:
            print("Computing minimum C_avg using {} score thresholds".format(len(thresholds)))
        # Collect labels and predictions for confusion matrix
        cm_labels = []
        cm_predictions = []
        for utt, y_true in trials_by_utt:
            if utt not in utt2scores:
                print("Warning: correct class for utterance '{}' listed in trials but it has no predicted scores, skipping".format(utt), file=sys.stderr)
                continue
            y_pred = utt2scores[utt]
            # Update using singleton batches
            cavg.update_state(
                tf.expand_dims(y_true, 0),
                tf.expand_dims(y_pred, 0)
            )
            cm_labels.append(tf.math.argmax(y_true))
            cm_predictions.append(tf.math.argmax(y_pred))
        # Evaluating the cavg result has a side effect of generating the argmin of the minimum cavg into cavg.min_index
        _ = cavg.result()
        min_threshold = thresholds[cavg.min_index].numpy()
        def print_metric(m):
            print("{:15s}\t{:.3f}".format(m.name + ":", m.result().numpy()))
        print("min C_avg at threshold {:.6f}".format(min_threshold))
        print_metric(cavg)
        # Now we know the threshold that minimizes C_avg and use the same threshold to compute all other metrics
        metrics = [
            M(langs, from_logits=from_logits, thresholds=min_threshold)
            for M in (AverageEqualErrorRate, AveragePrecision, AverageRecall)
        ]
        if args.verbosity:
            print("Computing rest of the metrics using threshold {:.6f}".format(min_threshold))
        for utt, y_true in trials_by_utt:
            if utt not in utt2scores:
                continue
            y_true_batch = tf.expand_dims(y_true, 0)
            y_pred = tf.expand_dims(utt2scores[utt], 0)
            for m in metrics:
                m.update_state(y_true_batch, y_pred)
        for avg_m in metrics:
            print_metric(avg_m)
        print("\nMetrics by target, using threshold {:.6f}".format(min_threshold))
        for avg_m in metrics:
            print(avg_m.name)
            for m in avg_m:
                print_metric(m)
        print("\nConfusion matrix")
        cm_labels = tf.cast(tf.stack(cm_labels), dtype=tf.int32)
        cm_predictions = tf.cast(tf.stack(cm_predictions), dtype=tf.int32)
        confusion_matrix = tf.math.confusion_matrix(cm_labels, cm_predictions, len(langs))
        print(langs)
        print(np.array_str(confusion_matrix.numpy()))

    def run(self):
        super().run()
        return self.evaluate()


class Util(E2EBase):
    tasks = (
        "get_cache_checksum",
    )

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        optional = parser.add_argument_group("util options")
        optional.add_argument("--get-cache-checksum",
            type=str,
            metavar="datagroup_key",
            help="For a given datagroup key, compute md5sum of config file in the same way as it would be computed when generating the filename for the features cache. E.g. for checking if the pipeline will be using the cache or start the feature extraction from scratch.")
        return parser

    def get_cache_checksum(self):
        datagroup_key = self.args.get_cache_checksum
        conf_json, conf_checksum = config_checksum(self.experiment_config, datagroup_key)
        if self.args.verbosity:
            print(10*'-' + " md5sum input begin " + 10*'-')
            print(conf_json, end='')
            print(10*'-' + "  md5sum input end  " + 10*'-')
        print("cache md5 checksum for datagroup key '{}' is:".format(datagroup_key))
        print(conf_checksum)

    def run(self):
        super().run()
        return self.run_tasks()


command_tree = [
    (E2E, [Train, Predict, Evaluate, Util]),
]
