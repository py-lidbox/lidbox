import collections
import hashlib
import itertools
import json
import os
import random
import sys
import time

import tensorflow as tf
import numpy as np

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

#TODO ensure input consistency, from config file, not some dynamically patched thing
def dict_checksum(d):
    json_str = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return json_str, hashlib.md5(json_str.encode("utf-8")).hexdigest()

def get_wav_config(conf, datagroup_key):
    return dict(conf["dataset"]["datagroups"][datagroup_key], sample_rate=conf["dataset"]["sample_rate"])

def accumulate_batch_size_counts(counter, t):
    return tf.tensor_scatter_nd_add(
        counter,
        tf.reshape(tf.shape(t[0])[0] - 1, (1, 1)),
        tf.constant([1]))

def count_batch_sizes(ds):
    max_batch_size = dataset[ds].reduce(
        tf.constant(-1, dtype=tf.int32),
        lambda max, t: tf.math.maximum(max, tf.shape(t[0])[0]))
    batch_size_counts = dataset[ds].reduce(
        tf.zeros(max_batch_size, dtype=tf.int32),
        accumulate_batch_size_counts)
    sorted_batch_sizes = tf.argsort(batch_size_counts, direction="DESCENDING")
    sorted_batch_size_counts = tf.gather(batch_size_counts, sorted_batch_sizes)
    return tf.stack(
        (tf.boolean_mask(sorted_batch_sizes, sorted_batch_size_counts > 0) + 1,
         tf.boolean_mask(sorted_batch_size_counts, sorted_batch_size_counts > 0)),
        axis=1)


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

    def extract_features(self, config, datagroup_key, copy_original_audio, trim_audio, debug_squeeze_last_dim):
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
                yaml_pprint(keys)
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
            feat = tf_data.extract_features_from_paths(
                config,
                get_wav_config(self.experiment_config, datagroup_key),
                paths,
                paths_meta,
                verbosity=args.verbosity,
                copy_original_audio=copy_original_audio,
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
                yaml_pprint(training_config[ds])
            ds_config = dict(training_config, **training_config[ds])
            del ds_config["train"], ds_config["validation"]
            summary_kwargs = dict(ds_config.get("dataset_logger", {}))
            debug_squeeze_last_dim = ds_config["input_shape"][-1] == 1
            datagroup_key = ds_config.pop("datagroup")
            extractor_ds = self.extract_features(
                feat_config,
                datagroup_key,
                summary_kwargs.get("copy_original_audio", False) or feat_config.get("voice_activity_detection", {}).get("match_feature_frames", False),
                summary_kwargs.pop("trim_audio", False),
                debug_squeeze_last_dim,
            )
            if ds_config.get("cache_features_in_tmp", False):
                features_cache_dir = "/tmp"
            else:
                features_cache_dir = os.path.join(self.cache_dir, "features")
            conf_json, conf_checksum = dict_checksum({
                "features": self.experiment_config["features"],
                "wav_config": get_wav_config(self.experiment_config, datagroup_key),
            })
            features_cache_path = os.path.join(
                features_cache_dir,
                self.experiment_config["dataset"]["key"],
                ds,
                feat_config["type"],
                conf_checksum,
            )
            self.make_named_dir(os.path.dirname(features_cache_path), "features cache")
            with open(features_cache_path + ".md5sum-input", "w") as f:
                print(conf_json, file=f, end='')
            if args.verbosity:
                print("Using cache for features: '{}'".format(features_cache_path))
            extractor_ds = extractor_ds.cache(filename=features_cache_path)
            if args.exhaust_dataset_iterator:
                if args.verbosity:
                    print("--exhaust-dataset-iterator given, now iterating once over the dataset iterator to fill the features cache.")
                # This forces the extractor_ds pipeline to be evaluated, and the features being serialized into the cache
                i = 0
                for i, x in enumerate(extractor_ds):
                    if args.verbosity > 1 and i % 2000 == 0:
                        print(i, "samples done")
                        if args.verbosity > 3:
                            print("sample", i, "shape is", tf.shape(x))
                if args.verbosity > 1:
                    print("all", i, "samples done")
            if args.verbosity > 2:
                print("Preparing dataset iterator for training")
            if "frames" in feat_config:
                print("'frames' key in feat_config does nothing, put it into the datagroup config under 'experiment'")
            dataset[ds] = tf_data.prepare_dataset_for_training(
                extractor_ds,
                ds_config,
                feat_config,
                label2onehot,
            )
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
            if args.debug_dataset:
                if args.verbosity:
                    print("--debug-dataset given, iterating over the dataset to gather stats")
                if args.verbosity > 1:
                    print("Counting all unique batch sizes in dataset")
                batch_size_counts = count_batch_sizes(dataset[ds])
                if args.verbosity:
                    print("Batch size counts:")
                tf.print(res, summarize=-1, output_stream=sys.stdout)
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
        if feat_config["type"] in ("melspectrogram", "logmelspectrogram", "mfcc"):
            assert "sample_rate" in self.experiment_config["dataset"], "dataset.sample_rate must be defined in the config file when feature type is '{}'".format(feat_config["type"])
            if "melspectrogram" not in feat_config:
                feat_config["melspectrogram"] = {}
            feat_config["melspectrogram"]["sample_rate"] = self.experiment_config["dataset"]["sample_rate"]
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
        keys = list(utt2path.keys())
        if args.file_limit:
            keys = keys[:args.file_limit]
            if args.verbosity > 3:
                print("Using utterance ids:")
                yaml_pprint(keys)
        int2label = self.experiment_config["dataset"]["labels"]
        label2int, OH = make_label2onehot(int2label)
        label2onehot = lambda label: OH[label2int.lookup(label)]
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

        copy_original_audio = feat_config.get("voice_activity_detection", {}).get("match_feature_frames", False)
        features = self.extract_features(
            feat_config,
            "test",
            copy_original_audio=copy_original_audio,
            trim_audio=False,
            debug_squeeze_last_dim=(ds_config["input_shape"][-1] == 1),
        )
        features = tf_data.prepare_dataset_for_training(
            features,
            ds_config,
            feat_config,
            label2onehot,
            copy_original_audio,
        )
        # drop meta wavs required only for vad
        features = features.map(lambda *t: t[:3])
        if ds_config.get("cache_features_in_tmp", False):
            features_cache_dir = "/tmp"
        else:
            features_cache_dir = os.path.join(self.cache_dir, "features")
        conf_json, conf_checksum = dict_checksum({
            "features": self.experiment_config["features"],
            "wav_config": get_wav_config(self.experiment_config, datagroup_key),
        })
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
            tf.print("Max score", max_score, "min score", min_score, output_stream=sys.stdout, summarize=-1)
        thresholds = tf.linspace(min_score, max_score, args.threshold_bins)
        if args.verbosity > 2:
            print("Score thresholds for language detection decisions:")
            tf.print(thresholds, summarize=5, output_stream=sys.stdout)
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
            metavar="datagroup",
            help="For a given datagroup key, compute md5sum of config file in the same way as it would be computed when generating the filename for the features cache. E.g. for checking if the pipeline will be using the cache or start the feature extraction from scratch.")
        return parser

    def get_cache_checksum(self):
        conf_json, conf_checksum = dict_checksum({
            "features": self.experiment_config["features"],
            "wav_config": get_wav_config(self.experiment_config, self.args.get_cache_checksum),
        })
        print(conf_checksum)
        if self.args.verbosity:
            print("Computed from json string '{}'".format(conf_json))

    def run(self):
        super().run()
        return self.run_tasks()


command_tree = [
    (E2E, [Train, Predict, Evaluate, Util]),
]
