from datetime import datetime
import collections
import os
import pprint
import random
import shutil
import sys

import sklearn.metrics
import tensorflow as tf

from speechbox.commands.base import Command, State, StatefulCommand, ExpandAbspath
import speechbox.dataset as dataset
import speechbox.tf_data as tf_data
import speechbox.models as models
import speechbox.preprocess.transformations as transformations
import speechbox.system as system
import speechbox.visualization as visualization


class E2E(Command):
    """TensorFlow pipeline for wavfile preprocessing, feature extraction and model training"""


def parse_space_separated(path):
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                yield l.split()

def make_label2onehot_fn(labels):
    labels_enum = tf.range(len(labels))
    # Label to int or one past last one if not found
    label2int = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(labels),
            tf.constant(labels_enum)),
        len(labels)
    )
    OH = tf.constant(tf.one_hot(labels_enum, len(labels)))
    return (lambda label: OH[label2int.lookup(label)])

def patch_feature_dim(config):
    if config["type"] == "mfcc":
        config["feature_dim"] = config["mfcc"]["num_coefs"]
    elif config["type"] in ("melspectrogram", "logmelspectrogram"):
        config["feature_dim"] = config["melspectrogram"]["num_mel_bins"]
    elif config["type"] == "spectrogram":
        config["feature_dim"] = config["spectrogram"] // 2 + 1
    elif config["type"] == "sparsespeech":
        assert "feature_dim" in config, "feature dimensions for sparsespeech decodings is equal to the number of mem entries (embedding output dim) and must be specified explicitly"
    else:
        assert False, "Error: unknown feature type '{}'".format(config["type"])
    return config


class Train(StatefulCommand):
    requires_state = State.none

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("options")
        optional.add_argument("--file-limit",
            type=int,
            help="Extract only up to this many files from the wavpath list (e.g. for debugging).")
        optional.add_argument("--shuffle-file-list",
            action="store_true",
            default=False,
            help="Shuffle wavpath list before loading wavs (e.g. for debugging, use TF shuffle buffers during training).")
        return parser

    def get_checkpoint_dir(self):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        return os.path.join(model_cache_dir, "checkpoints")

    def create_model(self, config):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        tensorboard_log_dir = os.path.join(model_cache_dir, "tensorboard", "logs")
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
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
        }
        self.make_named_dir(tensorboard_dir, "tensorboard")
        self.make_named_dir(checkpoint_dir, "checkpoints")
        if self.args.verbosity > 1:
            print("KerasWrapper callback parameters will be set to:")
            pprint.pprint(callbacks_kwargs)
            print()
        return models.KerasWrapper(self.model_id, config["model_definition"], **callbacks_kwargs)

    def extract_features(self, config, datagroup_key):
        args = self.args
        datagroup = self.experiment_config["dataset"]["datagroups"][datagroup_key]
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        if args.verbosity:
            print("Reading wav-files from '{}'".format(utt2path_path))
        utt2path = dict(row[:2] for row in parse_space_separated(utt2path_path))
        utt2label = dict(row[:2] for row in parse_space_separated(utt2label_path))
        # All utterance ids must be present in both files
        assert set(utt2path) == set(utt2label)
        paths = []
        paths_meta = []
        keys = list(utt2path.keys())
        if args.shuffle_file_list:
            if args.verbosity > 1:
                print("--shuffle-file-list given, shuffling utterance ids")
            random.shuffle(keys)
        if args.file_limit:
            if args.verbosity > 1:
                print("--file-limit set at {}, slicing utterance id list".format(args.file_limit))
            keys = keys[:args.file_limit]
        labels_set = set(self.experiment_config["dataset"]["labels"])
        num_dropped = 0
        for utt in keys:
            label = utt2label[utt]
            if label not in labels_set:
                num_dropped += 1
                continue
            paths.append(utt2path[utt])
            paths_meta.append((utt, label))
        if args.verbosity:
            print("Starting feature extraction for datagroup '{}' from {} files. Amount of files that were dropped because their label is not in the enabled labels list: {}".format(datagroup_key, len(paths), num_dropped))
            if "extracted_cache_dir" not in config:
                print("Warning: 'extracted_cache_dir' not specified in features config, features will be cached in memory.")
        extractor_ds, stats = tf_data.extract_features(config, paths, paths_meta)
        if args.verbosity > 1:
            print("Global dataset stats:")
            pprint.pprint(stats)
        return extractor_ds

    def train(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for training")
        training_config = self.experiment_config["experiment"]
        feat_config = self.experiment_config["features"]
        if args.verbosity > 1:
            print("Using model parameters:")
            pprint.pprint(training_config)
            print()
        if args.verbosity > 1:
            print("Using feature extraction parameters:")
            pprint.pprint(feat_config)
            print()
        feat_config = patch_feature_dim(feat_config)
        if feat_config["type"] in ("melspectrogram", "logmelspectrogram", "mfcc"):
            assert "sample_rate" in self.experiment_config["dataset"], "dataset.sample_rate must be defined in the config file when feature type is '{}'".format(feat_config["type"])
            if "melspectrogram" not in feat_config:
                feat_config["melspectrogram"] = {}
            feat_config["melspectrogram"]["sample_rate"] = self.experiment_config["dataset"]["sample_rate"]
        if "spectrogram" in feat_config:
            for k in ("frame_length", "frame_step"):
                if k in feat_config["spectrogram"]:
                    feat_config["voice_activity_detection"][k] = feat_config["spectrogram"][k]
        labels = self.experiment_config["dataset"]["labels"]
        label2onehot = make_label2onehot_fn(labels)
        if args.verbosity > 2:
            print("Generated onehot encoding:")
            for l in labels:
                l = tf.constant(l, dtype=tf.string)
                tf.print(l, label2onehot(l), summarize=-1, output_stream=sys.stdout)
        training_ds = self.extract_features(feat_config, training_config["training_datagroup"])
        validation_ds = self.extract_features(feat_config, training_config["validation_datagroup"])
        self.model_id = training_config["name"]
        model = self.create_model(training_config)
        training_ds = tf_data.prepare_dataset_for_training(training_ds, training_config, label2onehot)
        validation_ds = tf_data.prepare_dataset_for_training(validation_ds, training_config, label2onehot)
        summary_kwargs = training_config.get("monitor_training_input")
        if summary_kwargs:
            logdir = os.path.join(model.tensorboard.log_dir, "train")
            self.make_named_dir(logdir)
            train_ds_writer = tf.summary.create_file_writer(logdir)
            with train_ds_writer.as_default():
                training_ds = tf_data.attach_dataset_logger(training_ds, **summary_kwargs)
            logdir = os.path.join(model.tensorboard.log_dir, "validation")
            self.make_named_dir(logdir)
            validation_ds_writer = tf.summary.create_file_writer(logdir)
            with validation_ds_writer.as_default():
                validation_ds = tf_data.attach_dataset_logger(validation_ds, **summary_kwargs)
        if args.verbosity > 1:
            print("Compiling model")
        input_shape = next(iter(training_ds.take(1)))[0].shape
        if args.verbosity > 2:
            print("Full shape of the first sample in the training set is", input_shape)
        model.prepare(input_shape[1:], len(labels), training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoints = os.listdir(checkpoint_dir)
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, models.get_best_checkpoint(checkpoints, key="epoch"))
            if args.verbosity:
                print("Loading model weights from checkpoint file '{}'".format(checkpoint_path))
            model.load_weights(checkpoint_path)
        if args.verbosity:
            print("Starting training with model:")
            print(str(model))
            print()
        model.fit(training_ds, validation_ds, training_config)
        if args.verbosity:
            print("\nTraining finished\n")

    def run(self):
        super().run()
        return self.train()


command_tree = [
    (E2E, [Train]),
]
