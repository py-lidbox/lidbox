import collections
import itertools
import os
import pprint
import random
import sys
import time

import tensorflow as tf
import numpy as np

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
            feat, stats = tf_data.parse_sparsespeech_features(config, enc_path, feat_path, seg2utt, utt2label), {}
        else:
            feat, stats = tf_data.extract_features_from_paths(
                config,
                self.experiment_config["dataset"],
                paths,
                paths_meta,
                debug=args.debug_dataset,
                copy_original_audio=copy_original_audio,
                trim_audio=trim_audio,
                debug_squeeze_last_dim=debug_squeeze_last_dim,
            )
        return feat, stats

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
                copy_original_audio=summary_kwargs.get("copy_original_audio", False) or feat_config.get("voice_activity_detection", {}).get("match_feature_frames", False),
                trim_audio=summary_kwargs.pop("trim_audio", False),
                debug_squeeze_last_dim=debug_squeeze_last_dim,
            )
            if args.debug_dataset:
                print("Global dataset stats:")
                pprint.pprint(dict(stats))
                for key in ("global_min", "global_max"):
                    tf.debugging.assert_all_finite(stats["features"][key], "Some feature dimension is missing values")
            else:
                tmp_features_cache = os.path.join("/tmp", self.experiment_config["dataset"]["key"], ds, feat_config["type"])
                if args.verbosity > 1:
                    print("Caching feature extractor dataset to '{}'".format(tmp_features_cache))
                self.make_named_dir(os.path.dirname(tmp_features_cache), "features cache")
                if args.verbosity:
                    print("Using cache for features: '{}'".format(tmp_features_cache))
                extractor_ds = extractor_ds.cache(filename=tmp_features_cache)
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
            if args.inspect_dataset and summary_kwargs:
                logdir = os.path.join(os.path.dirname(model.tensorboard.log_dir), "dataset", ds)
                if args.verbosity > 1:
                    print("Datagroup '{}' has a dataset logger defined. We will iterate over the dataset once to create TensorBoard summaries of the input data into '{}'.".format(ds, logdir))
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


class Predict(E2EBase):
    """
    Use a trained model to produce likelihoods for all target languages from all utterances in the test set.
    Writes all predictions as scores and information about the target and non-target languages into the cache dir.
    """

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("predict options")
        optional.add_argument("--score-precision", type=int, default=3)
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
        if not args.trials:
            args.trials = os.path.join(self.cache_dir, "predictions", "trials")
        if not args.scores:
            args.scores = os.path.join(self.cache_dir, "predictions", "scores")
        self.make_named_dir(os.path.dirname(args.trials))
        self.make_named_dir(os.path.dirname(args.scores))
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
            latest_checkpoint = models.get_best_checkpoint(checkpoints, key="epoch")
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
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
        )
        tmp_features_cache = os.path.join("/tmp", self.experiment_config["dataset"]["key"], ds, feat_config["type"])
        if args.verbosity > 1:
            print("Caching feature extractor dataset to '{}'".format(tmp_features_cache))
        self.make_named_dir(os.path.dirname(tmp_features_cache), "features cache")
        features = features.cache(tmp_features_cache)
        first = list(features.take(1))
        assert first, "feature extraction failed, 'features' tf.data.Dataset does not contain any elements"
        if args.verbosity > 2:
            print("Peeking first element in the features dataset iterator:", first)
        # Gather utterance ids, this also causes the extraction pipeline to be evaluated
        utterance_ids = [meta[0][0].numpy().decode("utf-8") for _, meta in features]
        features = features.map(lambda feats, *meta: feats)
        if "batch_size" in ds_config:
            if args.verbosity > 1:
                print("Predicting in batches, batching to {}".format(ds_config["batch_size"]))
            features = features.unbatch().batch(ds_config["batch_size"])
        if args.verbosity:
            print("Features extracted, writing target and non-target language information for each utterance to '{}'.".format(args.trials))
        with open(args.trials, "w") as trials_f:
            for utt, target in utt2label.items():
                for lang in int2label:
                    print(lang, utt, "target" if target == lang else "nontarget", file=trials_f)
        if args.verbosity:
            print("Starting prediction")
        predictions = model.predict(features.prefetch(tf.data.experimental.AUTOTUNE))
        num_predictions = 0
        with open(args.scores, "w") as scores_f:
            print(*int2label, file=scores_f)
            with np.printoptions(precision=args.score_precision, suppress=True, floatmode='fixed'):
                for num_predictions, (utt, pred) in enumerate(zip(utterance_ids, predictions), start=1):
                    # Numpy array as string without square brackets and comma delimiters
                    scores_str = np.array_str(pred)[1:-1]
                    print(utt, scores_str, file=scores_f)
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
        optional.add_argument("--log-to-prob", action="store_true", default=False)
        return parser

    def evaluate(self):
        args = self.args
        if not args.trials:
            args.trials = os.path.join(self.cache_dir, "predictions", "trials")
        if not args.scores:
            args.scores = os.path.join(self.cache_dir, "predictions", "scores")
        if args.verbosity > 1:
            print("Evaluating minimum average detection cost using trials '{}' and scores '{}'".format(args.trials, args.scores))
        score_lines = list(parse_space_separated(args.scores))
        langs = score_lines[0]
        lang2int = {l: i for i, l in enumerate(langs)}
        utt2scores = {utt: tf.constant([float(s) for s in scores], dtype=tf.float32) for utt, *scores in score_lines[1:]}
        if args.verbosity > 1:
            print("Parsed scores for {} utterances".format(len(utt2scores)))
        if args.log_to_prob:
            print("Converting log likelihood scores into probabilities")
            utt2scores = {utt: tf.math.exp(scores) for utt, scores in utt2scores.items()}
            for utt, scores in utt2scores.items():
                one = tf.constant(1.0, dtype=tf.float32)
                tf.debugging.assert_near(tf.reduce_sum(scores), one, message="failed to convert log likelihoods to probabilities, the probabilities of predictions for utterance '{}' does not sum to 1")
        if args.verbosity > 1:
            print("Generating {} threshold bins".format(args.threshold_bins))
        assert args.threshold_bins > 0
        max_score = tf.constant(-float("inf"), dtype=tf.float32)
        min_score = tf.constant(float("inf"), dtype=tf.float32)
        for utt, scores in utt2scores.items():
            max_score = tf.math.maximum(max_score, tf.math.reduce_max(scores))
            min_score = tf.math.minimum(min_score, tf.math.reduce_min(scores))
        if args.verbosity > 2:
            tf.print("Max score", max_score, "min score", min_score, output_stream=sys.stdout, summarize=-1)
        thresholds = tf.linspace(min_score, max_score, args.threshold_bins)
        if args.verbosity > 2:
            print("Score thresholds for language detection decisions:")
            tf.print(thresholds, output_stream=sys.stdout)
        # First do C_avg to get the best threshold
        cavg = AverageDetectionCost(langs, theta_det=list(thresholds.numpy()))
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
            AverageEqualErrorRate(langs, thresholds=min_threshold),
            AveragePrecision(langs, thresholds=min_threshold),
            AverageRecall(langs, thresholds=min_threshold),
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
        tf.print(confusion_matrix, summarize=-1, output_stream=sys.stdout)

    def run(self):
        super().run()
        return self.evaluate()


command_tree = [
    (E2E, [Train, Predict, Evaluate]),
]
