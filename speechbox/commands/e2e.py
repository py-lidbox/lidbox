from datetime import datetime
import collections
import os
import pprint
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
    label2int = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(labels),
            tf.constant(labels_enum)),
        len(labels)
    )
    OH = tf.one_hot(labels_enum, len(labels))
    return (lambda label: OH[label2int.lookup(label)])

class Train(StatefulCommand):
    requires_state = State.none

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
        for utt, path in utt2path.items():
            paths.append(path)
            paths_meta.append((utt, utt2label[utt]))
        cache_path = os.path.join(self.cache_dir, "tf_data", datagroup_key)
        self.make_named_dir(os.path.dirname(cache_path), "tf.data.Dataset cache")
        if args.verbosity:
            print("Starting feature extraction for datagroup '{}'".format(datagroup_key))
        extractor_ds, stats = tf_data.extract_features(config, paths, paths_meta)
        if args.verbosity > 1:
            print("Global dataset stats:")
            pprint.pprint(stats)
        # return extractor_ds.cache(cache_path)
        return extractor_ds

    @staticmethod
    def patch_feature_dim(config):
        if config["type"] == "mfcc":
            config["feature_dim"] = config["mfcc"]["num_coefs"]
        elif config["type"] == "melspectrogram":
            config["feature_dim"] = config["melspectrogram"]["num_mel_bins"]
            config["mfcc"] = None
        elif config["type"] == "logmelspectrogram":
            config["feature_dim"] = config["melspectrogram"]["num_mel_bins"]
            config["mfcc"] = None
            config["logmel"] = True
        elif config["type"] == "spectrogram":
            config["feature_dim"] = config["spectrogram"] // 2 + 1
            config["melspectrogram"] = config["mfcc"] = None
        else:
            print("Error: unknown feature type '{}'".format(config["type"]))
            config = None
        return config

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
        feat_config = self.patch_feature_dim(feat_config)
        training_ds = self.extract_features(feat_config, training_config["training_datagroup"])
        validation_ds = self.extract_features(feat_config, training_config["validation_datagroup"])
        self.model_id = training_config["name"]
        model = self.create_model(training_config)
        if training_config.get("monitor_training_input", False):
            metrics_dir, training_ds = model.enable_dataset_logger("training", training_ds)
            self.make_named_dir(metrics_dir)
        if args.verbosity > 1:
            print("Compiling model")
        labels = self.experiment_config["dataset"]["labels"]
        input_shape = (feat_config["min_sequence_length"], feat_config["feature_dim"])
        model.prepare(input_shape, len(labels), training_config)
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
        label2onehot = make_label2onehot_fn(labels)
        training_ds = tf_data.prepare_dataset_for_training(training_ds, training_config, label2onehot)
        validation_ds = tf_data.prepare_dataset_for_training(validation_ds, training_config, label2onehot)
        model.fit(training_ds, validation_ds, training_config)
        if args.verbosity:
            print("\nTraining finished\n")

    def run(self):
        super().run()
        # metrics_dir = os.path.join(self.cache_dir, "trace")
        # with tf.summary.create_file_writer(metrics_dir).as_default():
            # tf.summary.trace_on(profiler=True)
        ret = self.train()
        return ret


command_tree = [
    (E2E, [Train]),
]
