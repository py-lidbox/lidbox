import collections
import datetime
import hashlib
import importlib
import itertools
import json
import os
import random
import sys
import time

import kaldiio
import numpy as np
import tensorflow as tf

from .. import (
    models,
    parse_space_separated,
    system,
    tf_data,
    tf_util,
    yaml_pprint,
)
from .base import BaseCommand, Command, ExpandAbspath


def now_str(date=False):
    return str(datetime.datetime.now() if date else int(time.time()))


class E2E(BaseCommand):
    """TensorFlow pipeline for wavfile preprocessing, feature extraction and sound classification model training"""
    pass


class E2EBase(Command):

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("options")
        optional.add_argument("--file-limit",
            type=int,
            help="Extract only up to this many files from the wavpath list (e.g. for debugging). Default is no limit.")
        return parser

    def get_checkpoint_dir(self):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        return os.path.join(model_cache_dir, "checkpoints")

    def create_model(self, config, skip_training=False):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        tensorboard_log_dir = os.path.join(model_cache_dir, "tensorboard", "logs")
        tensorboard_dir = os.path.join(tensorboard_log_dir, now_str())
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
        optional.add_argument("--shuffle-utt2path",
            action="store_true",
            default=False,
            help="Override utt2path shuffling")
        optional.add_argument("--exhaust-dataset-iterator",
            action="store_true",
            default=False,
            help="Explictly iterate once over the feature extractor tf.data.Dataset object in order to evaluate the feature extraction pipeline and fill the feature cache on disk. Using this with --skip-training allows you to extract features on multiple CPUs without needing a GPU.")
        optional.add_argument("--dataset-config",
            type=str,
            action=ExpandAbspath,
            help="Path to a yaml-file containing a list of datasets.")
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
        if args.dataset_config:
            dataset_config = system.load_yaml(args.dataset_config)
            self.experiment_config["datasets"] = [d for d in dataset_config if d["key"] in self.experiment_config["datasets"]]
        labels = sorted(set(l for d in self.experiment_config["datasets"] for l in d["labels"]))
        label2int, OH = tf_util.make_label2onehot(labels)
        def label2onehot(label):
            return OH[label2int.lookup(label)]
        if args.verbosity > 2:
            print("Generating onehot encoding from labels:", ', '.join(labels))
            print("Generated onehot encoding as tensors:")
            for l in labels:
                l = tf.constant(l, dtype=tf.string)
                tf_util.tf_print(l, "\t", label2onehot(l))
        self.model_id = training_config["name"]
        model = self.create_model(dict(training_config), args.skip_training)
        if args.verbosity > 1:
            print("Preparing model")
        model.prepare(labels, training_config)
        if args.verbosity:
            print("Using model:\n{}".format(str(model)))
        dataset = {}
        for ds in ("train", "validation"):
            if args.verbosity > 2:
                print("Dataset config for '{}'".format(ds))
                yaml_pprint(training_config[ds])
            ds_config = dict(training_config, **training_config[ds])
            del ds_config["train"], ds_config["validation"]
            summary_kwargs = dict(ds_config.get("dataset_logger", {}))
            datagroup_key = ds_config.pop("datagroup")
            tf_util_kwargs = {
                    "verbosity": args.verbosity,
                    "force_shuffle_utt2path": args.shuffle_utt2path,
                    "file_limit": args.file_limit}
            if "features_cache" not in ds_config:
                if args.verbosity:
                    print("features_cache not defined in config, will not cache extracted features")
                extractor_ds = tf_util.extract_features(
                    self.experiment_config["datasets"],
                    feat_config,
                    datagroup_key,
                    **tf_util_kwargs)
            else:
                if args.verbosity:
                    print("features_cache defined in config, extracted features will be cached or existing features will be loaded from cache")
                conf_json, conf_checksum = system.config_checksum(self.experiment_config, datagroup_key)
                if args.verbosity > 2:
                    print("Config md5 checksum '{}' computed from json string:".format(conf_checksum))
                    print(conf_json)
                tf_util_args = ds_config, self.experiment_config, datagroup_key, self.cache_dir
                tf_util_kwargs["conf_json"] = conf_json
                tf_util_kwargs["conf_checksum"] = conf_checksum
                cache_type = ds_config["features_cache"]["type"]
                if cache_type == "tf_data_cache":
                    extractor_ds = tf_util.extract_features_with_cache(
                            *tf_util_args,
                            **tf_util_kwargs)
                elif cache_type == "tfrecords":
                    extractor_ds = tf_util.extract_features_with_tfrecords(
                            *tf_util_args,
                            **tf_util_kwargs)
                else:
                    print("error: invalid features cache type '{}'".format(cache_type))
                    return 1
            extractor_ds = extractor_ds.unbatch()
            if args.exhaust_dataset_iterator:
                if args.verbosity:
                    print("--exhaust-dataset-iterator given, now iterating once over the dataset iterator to fill the features cache.")
                # This forces the extractor_ds pipeline to be evaluated, and the features being serialized into the cache
                i = 0
                if args.verbosity > 1:
                    print(now_str(date=True), "- 0 samples done")
                for i, (feats, *meta) in enumerate(extractor_ds.as_numpy_iterator(), start=1):
                    if args.verbosity > 1 and i % 10000 == 0:
                        print(now_str(date=True), "-", i, "samples done")
                    if args.verbosity > 3:
                        tf_util.tf_print("sample:", i, "features shape:", tf.shape(feats), "metadata:", *meta)
                if args.verbosity > 1:
                    print(now_str(date=True), "- all", i, "samples done")
            dataset[ds] = tf_data.prepare_dataset_for_training(
                extractor_ds,
                ds_config,
                feat_config,
                label2onehot,
                self.model_id,
                conf_checksum=conf_checksum,
                verbosity=args.verbosity,
            )
            if args.debug_dataset:
                if args.verbosity:
                    print("--debug-dataset given, iterating over the dataset to gather stats")
                if args.verbosity > 1:
                    print("Counting all unique dim sizes of elements at index 0 in dataset")
                for axis, size_counts in enumerate(tf_util.count_dim_sizes(dataset[ds], 0, len(ds_config["input_shape"]) + 1)):
                    print("axis {}\n[count size]:".format(axis))
                    tf_util.tf_print(size_counts, summarize=10)
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
                        with writer.as_default():
                            logged_dataset = tf_data.attach_dataset_logger(dataset[ds], feat_config["type"], **summary_kwargs)
                            if args.verbosity:
                                print("Dataset logger attached to '{0}' dataset iterator, now exhausting the '{0}' dataset logger iterator once to write TensorBoard summaries of model input data".format(ds))
                            i = 0
                            max_outputs = summary_kwargs.get("max_outputs", 10)
                            for i, (samples, labels, *meta) in enumerate(logged_dataset.as_numpy_iterator()):
                                if args.verbosity > 1 and i % (2000//ds_config.get("batch_size", 1)) == 0:
                                    print(i, "batches done")
                                if args.verbosity > 3:
                                    tf_util.tf_print(
                                            "batch:", i,
                                            "utts", meta[0][:max_outputs],
                                            "samples shape:", tf.shape(samples),
                                            "onehot shape:", tf.shape(labels),
                                            "wav.audio.shape", meta[1].audio.shape,
                                            "wav.sample_rate[0]", meta[1].sample_rate[0])
                            if args.verbosity > 1:
                                print(i, "batches done")
                            del logged_dataset
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
            print("\nStarting training")
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
        history_cache_dir = os.path.join(self.cache_dir, self.model_id, "history")
        now_s = now_str()
        for name, epoch_vals in history.history.items():
            history_file = os.path.join(history_cache_dir, now_s, name)
            self.make_named_dir(os.path.dirname(history_file), "training history")
            with open(history_file, "w") as f:
                for epoch, val in enumerate(epoch_vals, start=1):
                    print(epoch, val, file=f)
            if args.verbosity > 1:
                print("wrote history file '{}'".format(history_file))

    def run(self):
        super().run()
        return self.train()


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
        conf_json, conf_checksum = system.config_checksum(self.experiment_config, datagroup_key)
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
    (E2E, [Train, Util]),
]
