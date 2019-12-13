import collections
import os
import pprint
import random
import sys
import time

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
    """TensorFlow pipeline for wavfile preprocessing, feature extraction and sound classification model training"""
    pass


def parse_space_separated(path):
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                yield l.split()

# TODO move towards integer labels and sparse categorical cross entropy to drop all onehot labels
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


class E2EBase(StatefulCommand):
    requires_state = State.none

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("options")
        optional.add_argument("--file-limit",
            type=int,
            help="Extract only up to this many files from the wavpath list (e.g. for debugging).")
        optional.add_argument("--debug-dataset",
            action="store_true",
            default=False,
            help="Gather statistics and shapes of elements in the datasets during feature extraction. This adds several linear passes over the datasets.")
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
        }
        if not skip_training:
            self.make_named_dir(tensorboard_dir, "tensorboard")
            self.make_named_dir(checkpoint_dir, "checkpoints")
        if self.args.verbosity > 1:
            print("KerasWrapper callback parameters will be set to:")
            pprint.pprint(callbacks_kwargs)
            print()
        return models.KerasWrapper(self.model_id, config["model_definition"], **callbacks_kwargs)

    def extract_features(self, config, datagroup_key, copy_original_audio, trim_audio, debug_squeeze_last_dim):
        args = self.args
        datagroup = self.experiment_config["dataset"]["datagroups"][datagroup_key]
        if args.verbosity > 2:
            print("Extracting features from datagroup '{}' with config".format(datagroup_key))
            pprint.pprint(config)
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
        keys = list(utt2path.keys())
        if datagroup.get("shuffle_utt2path", False):
            if args.verbosity > 1:
                print("Shuffling utterance ids, all wavpaths in the utt2path list will be processed in random order.")
            random.shuffle(keys)
        else:
            if args.verbosity > 1:
                print("Not shuffling utterance ids, all wavs will be processed in order of the utt2path list.")
        if args.file_limit:
            if args.verbosity > 1:
                print("--file-limit set at {0}, using at most {0} utterances from the utterance id list, starting at the beginning of utt2path".format(args.file_limit))
            keys = keys[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                pprint.pprint(keys)
        labels_set = set(self.experiment_config["dataset"]["labels"])
        num_dropped = collections.Counter()
        paths = []
        paths_meta = []
        for utt in keys:
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
            if args.debug_dataset:
                print("--debug-dataset given, stats will be gathered from extracted features by doing several linear passes over the dataset before training starts. Features will not be cached.")
            if args.verbosity > 3:
                print("Using paths:")
                for path, (utt, label) in zip(paths, paths_meta):
                    print(utt, path, label)
        num_cores = len(os.sched_getaffinity(0))
        if args.verbosity:
            print("This process has been assigned {0} physical cores, using {0} parallel calls during feature extraction".format(num_cores))
        return tf_data.extract_features_from_paths(
            config,
            self.experiment_config["dataset"],
            paths,
            paths_meta,
            debug=args.debug_dataset,
            copy_original_audio=copy_original_audio,
            trim_audio=trim_audio,
            debug_squeeze_last_dim=debug_squeeze_last_dim,
            num_cores=num_cores,
        )


class Train(E2EBase):

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("options")
        optional.add_argument("--skip-training",
            action="store_true",
            default=False)
        optional.add_argument("--inspect-dataset",
            action="store_true",
            default=False,
            help="Iterate once over the whole dataset before training, writing TensorBoard summaries of the data. This requires the dataset_logger key to be defined in the config file.")
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
            pprint.pprint(training_config)
            print()
        if args.verbosity > 1:
            print("Using feature extraction parameters:")
            pprint.pprint(feat_config)
            print()
        if feat_config["type"] in ("melspectrogram", "logmelspectrogram", "mfcc"):
            assert "sample_rate" in self.experiment_config["dataset"], "dataset.sample_rate must be defined in the config file when feature type is '{}'".format(feat_config["type"])
            if "melspectrogram" not in feat_config:
                feat_config["melspectrogram"] = {}
            feat_config["melspectrogram"]["sample_rate"] = self.experiment_config["dataset"]["sample_rate"]
        labels = self.experiment_config["dataset"]["labels"]
        label2int, OH = make_label2onehot(labels)
        label2onehot = lambda label: OH[label2int.lookup(label)]
        if args.verbosity > 2:
            print("Generating onehot encoding from labels:", ', '.join(labels))
            print("Generated onehot encoding as tensors:")
            for l in labels:
                l = tf.constant(l, dtype=tf.string)
                tf.print(l, "\t", label2onehot(l), summarize=-1, output_stream=sys.stdout)
        self.model_id = training_config["name"]
        model = self.create_model(dict(training_config), args.skip_training)
        dataset = {}
        for ds in ("train", "validation"):
            if args.verbosity > 2:
                print("Dataset config for '{}'".format(ds))
                pprint.pprint(training_config[ds])
            ds_config = dict(training_config, **training_config[ds])
            del ds_config["train"], ds_config["validation"]
            summary_kwargs = dict(ds_config.get("dataset_logger", {}))
            debug_squeeze_last_dim = ds_config["input_shape"][-1] == 1
            datagroup_key = ds_config.pop("datagroup")
            extractor_ds, stats = self.extract_features(
                feat_config,
                datagroup_key,
                copy_original_audio=summary_kwargs.get("copy_original_audio", False),
                trim_audio=summary_kwargs.pop("trim_audio", False),
                debug_squeeze_last_dim=debug_squeeze_last_dim,
            )
            if args.debug_dataset:
                print("Global dataset stats:")
                pprint.pprint(dict(stats))
                for key in ("global_min", "global_max"):
                    tf.debugging.assert_all_finite(stats["features"][key], "Some feature dimension is missing values")
            else:
                features_cache_name = feat_config.get("name", feat_config["type"])
                features_cache = os.path.join(self.cache_dir, "features", features_cache_name, datagroup_key)
                self.make_named_dir(os.path.dirname(features_cache), "tf.data.Dataset features cache")
                if args.verbosity:
                    print("Using cache for features: '{}'".format(features_cache))
                extractor_ds = extractor_ds.cache(filename=features_cache)
            if args.exhaust_dataset_iterator:
                if args.verbosity:
                    print("--exhaust-dataset-iterator given, now iterating once over the dataset iterator to fill the features cache.")
                # This forces the extractor_ds pipeline to be evaluated, and the features being serialized into the cache
                for i, x in enumerate(extractor_ds):
                    if args.verbosity > 1 and i % 2000 == 0:
                        print(i, "samples done")
                        if args.verbosity > 3:
                            print("sample", i, "shape is", tf.shape(x))
            if args.verbosity > 2:
                print("Preparing dataset iterator for training")
            dataset[ds] = tf_data.prepare_dataset_for_training(
                extractor_ds,
                ds_config,
                feat_config,
                label2onehot,
            )
            if args.inspect_dataset:
                if not summary_kwargs:
                    print("Error: --inspect-dataset given but dataset_logger not defined in the config file, unable to attach dataset logger for inspecting dataset.", file=sys.stderr)
                    return 1
                logdir = os.path.join(os.path.dirname(model.tensorboard.log_dir), "dataset", ds)
                if args.verbosity > 1:
                    print("Datagroup '{}' has a dataset logger defined. We will iterate over the dataset once to create TensorBoard summaries of the input data into '{}'.".format(ds, logdir))
                self.make_named_dir(logdir)
                writer = tf.summary.create_file_writer(logdir)
                summary_kwargs["debug_squeeze_last_dim"] = debug_squeeze_last_dim
                with writer.as_default():
                    num_cores = len(os.sched_getaffinity(0))
                    logged_dataset = tf_data.attach_dataset_logger(dataset[ds], feat_config["type"], num_cores=num_cores, **summary_kwargs)
                    if args.verbosity:
                        print("Dataset logger attached to '{0}' dataset iterator, now exhausting the '{0}' dataset logger iterator once to write TensorBoard summaries of model input data".format(ds))
                    for i, elem in enumerate(logged_dataset):
                        if args.verbosity > 1 and i % (2000//ds_config.get("batch_size", 1)) == 0:
                            print(i, "batches done")
                    if args.verbosity > 1:
                        print(i, "batches done")
        if args.verbosity > 1:
            print("Compiling model")
        model.prepare(len(labels), training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoints = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, models.get_best_checkpoint(checkpoints, key="epoch"))
            if args.verbosity:
                print("Loading model weights from checkpoint file '{}'".format(checkpoint_path))
            model.load_weights(checkpoint_path)
        if args.verbosity:
            print("\nStarting training with:")
            print(str(model))
            print()
        if args.skip_training:
            print("--skip-training given, will not call model.fit")
            return
        model.fit(dataset["train"], dataset["validation"], training_config)
        if args.verbosity:
            print("\nTraining finished\n")

    def run(self):
        super().run()
        return self.train()


class Evaluate(E2EBase):

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("evaluate options")
        optional.add_argument("--checkpoint",
            type=str,
            help="Specify which Keras checkpoint to load model weights from, instead of using the most recent one.")
        optional.add_argument("--confusion-matrix-path",
            type=str,
            action=ExpandAbspath,
            metavar="./cache/evaluations/cm.png",
            help="Path for the confusion matrix output image. Given directly to plt.savefig so you can also choose the filetype with this arg.")
        optional.add_argument("--eval-result-dir",
            type=str,
            action=ExpandAbspath,
            metavar="./cache/evaluations",
            help="Write evaluation results into this directory.")
        return parser

    def evaluate(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for evaluation")
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
        if feat_config["type"] in ("melspectrogram", "logmelspectrogram", "mfcc"):
            assert "sample_rate" in self.experiment_config["dataset"], "dataset.sample_rate must be defined in the config file when feature type is '{}'".format(feat_config["type"])
            if "melspectrogram" not in feat_config:
                feat_config["melspectrogram"] = {}
            feat_config["melspectrogram"]["sample_rate"] = self.experiment_config["dataset"]["sample_rate"]
        labels = self.experiment_config["dataset"]["labels"]
        label2int, OH = make_label2onehot(labels)
        label2onehot = lambda label: OH[label2int.lookup(label)]
        if args.verbosity > 2:
            print("Generating onehot encoding from labels:", ', '.join(labels))
            print("Generated onehot encoding as tensors:")
            for l in labels:
                l = tf.constant(l, dtype=tf.string)
                tf.print(l, "\t", label2onehot(l), summarize=-1, output_stream=sys.stdout)
        self.model_id = training_config["name"]
        model = self.create_model(dict(training_config), skip_training=True)
        if args.verbosity > 1:
            print("Compiling model")
        model.prepare(len(labels), training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoints = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
        if not checkpoints:
            print("Error: Cannot evaluate with a model that has no checkpoints, i.e. is not trained.")
            return 1
        latest_checkpoint = models.get_best_checkpoint(checkpoints, key="epoch")
        checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint or latest_checkpoint)
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
            pprint.pprint(training_config[ds])
        ds_config = dict(training_config, **training_config[ds])
        del ds_config["train"], ds_config["validation"]
        if "dataset_logger" in ds_config and args.verbosity:
            print("Warning: dataset_logger in the test datagroup has no effect.")
        datagroup = self.experiment_config["dataset"]["datagroups"][ds_config.pop("datagroup")]
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        utt2path = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2path_path)
        )
        utt2label = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2label_path)
        )
        keys = list(utt2path.keys())
        if args.file_limit:
            keys = keys[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                pprint.pprint(keys)
        int2label = self.experiment_config["dataset"]["labels"]
        labels_set = set(int2label)
        print("utt, pred, real, likelihoods")
        for utt in keys:
            label = utt2label[utt]
            if label not in labels_set:
                continue
            extractor_ds, _ = tf_data.extract_features_from_paths(
                feat_config,
                self.experiment_config["dataset"],
                [utt2path[utt]],
                [(utt, label)],
                debug=False,
                copy_original_audio=False,
                num_cores=len(os.sched_getaffinity(0)),
            )
            feature_frames = tf_data.prepare_dataset_for_training(
                extractor_ds,
                ds_config,
                feat_config,
                label2onehot,
            )
            pred = model.predict(feature_frames)
            mean = pred.mean(axis=0)
            print(utt, int2label[mean.argmax()], label, *["{:.3f}".format(x) for x in mean], sep="\t")

    def run(self):
        super().run()
        return self.evaluate()


class Predict(E2EBase):

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("evaluate options")
        optional.add_argument("--checkpoint",
            type=str,
            help="Specify which Keras checkpoint to load model weights from, instead of using the most recent one.")
        return parser

    def predict(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for prediction")
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
        if feat_config["type"] in ("melspectrogram", "logmelspectrogram", "mfcc"):
            assert "sample_rate" in self.experiment_config["dataset"], "dataset.sample_rate must be defined in the config file when feature type is '{}'".format(feat_config["type"])
            if "melspectrogram" not in feat_config:
                feat_config["melspectrogram"] = {}
            feat_config["melspectrogram"]["sample_rate"] = self.experiment_config["dataset"]["sample_rate"]
        self.model_id = training_config["name"]
        model = self.create_model(dict(training_config), skip_training=True)
        if args.verbosity > 1:
            print("Compiling model")
        labels = self.experiment_config["dataset"]["labels"]
        model.prepare(len(labels), training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoints = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
        if not checkpoints:
            print("Error: Cannot evaluate with a model that has no checkpoints, i.e. is not trained.")
            return 1
        latest_checkpoint = models.get_best_checkpoint(checkpoints, key="epoch")
        checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint or latest_checkpoint)
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
            pprint.pprint(training_config[ds])
        ds_config = dict(training_config, **training_config[ds])
        del ds_config["train"], ds_config["validation"]
        if args.verbosity and "dataset_logger" in ds_config:
            print("Warning: dataset_logger in the test datagroup has no effect.")
        datagroup = self.experiment_config["dataset"]["datagroups"][ds_config.pop("datagroup")]
        utt2path_path = os.path.join(datagroup["path"], datagroup.get("utt2path", "utt2path"))
        utt2label_path = os.path.join(datagroup["path"], datagroup.get("utt2label", "utt2label"))
        utt2path = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2path_path)
        )
        utt2label = collections.OrderedDict(
            row[:2] for row in parse_space_separated(utt2label_path)
        )
        keys = list(utt2path.keys())
        if args.file_limit:
            keys = keys[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                pprint.pprint(keys)
        int2label = self.experiment_config["dataset"]["labels"]
        labels_set = set(int2label)
        paths = []
        paths_meta = []
        for utt in keys:
            label = utt2label[utt]
            if label not in labels_set:
                continue
            paths.append(utt2path[utt])
            paths_meta.append((utt, label))
        if args.verbosity:
            print("Extracting test set features for prediction")
        features = tf_data.extract_features_for_prediction(
            feat_config,
            self.experiment_config["dataset"],
            paths,
            paths_meta,
            num_cores=len(os.sched_getaffinity(0)),
        )
        for feats, meta in features:
            utt, lang = meta[0], meta[1]
            pred = model.predict(features)
            tf.print(utt, lang, tf.shape(feats), int2label[pred.argmax()], pred, summarize=-1, output_stream=sys.stdout)

    def run(self):
        super().run()
        return self.predict()


command_tree = [
    (E2E, [Train, Evaluate, Predict]),
]
