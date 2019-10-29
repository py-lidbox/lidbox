from datetime import datetime
import collections
import os
import pprint
import shutil
import sys

import sklearn.metrics

from speechbox.commands.base import Command, State, StatefulCommand, ExpandAbspath
import speechbox.dataset as dataset
import speechbox.models as models
import speechbox.preprocess.transformations as transformations
import speechbox.system as system
import speechbox.visualization as visualization


class Model(Command):
    """Model training and evaluation"""


class Train(StatefulCommand):
    """Train a TensorFlow model"""
    requires_state = State.has_features

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        optional = parser.add_argument_group("training options")
        optional.add_argument("--imbalanced-labels",
            action="store_true",
            help="Apply weighting on imbalanced labels during training by using a pre-calculated feature distribution.")
        optional.add_argument("--reset-tensorboard",
            action="store_true",
            help="Delete tensorboard directory from previous runs for this model.")
        optional.add_argument("--reset-checkpoints",
            action="store_true",
            help="Delete checkpoints from previous runs for this model.")
        return parser

    def get_checkpoint_dir(self):
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        return os.path.join(model_cache_dir, "checkpoints")

    def create_model(self, config):
        args = self.args
        model_cache_dir = os.path.join(self.cache_dir, self.model_id)
        tensorboard_log_dir = os.path.join(model_cache_dir, "tensorboard", "logs")
        if args.reset_tensorboard and os.path.isdir(tensorboard_log_dir):
            if args.verbosity:
                print("Clearing tensorboard directory '{}'".format(tensorboard_log_dir))
            shutil.rmtree(tensorboard_log_dir)
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard_dir = os.path.join(tensorboard_log_dir, now_str)
        default_tensorboard_config = {
            "log_dir": tensorboard_dir,
            "profile_batch": 0,
            "histogram_freq": 1,
        }
        if args.debug:
            default_tensorboard_config.update({
                "update_freq": "batch",
                "histogram_freq": 1,
            })
        tensorboard_config = dict(default_tensorboard_config, **config.get("tensorboard", {}))
        checkpoint_dir = self.get_checkpoint_dir()
        if args.reset_checkpoints and os.path.isdir(checkpoint_dir):
            if args.verbosity:
                print("Clearing checkpoint directory '{}'".format(checkpoint_dir))
            shutil.rmtree(checkpoint_dir)
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
        if args.verbosity > 1:
            print("KerasWrapper callback parameters will be set to:")
            pprint.pprint(callbacks_kwargs)
            print()
        return models.KerasWrapper(self.model_id, config["model_definition"], **callbacks_kwargs)

    def train(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for training")
        if not self.state_data_ok():
            return 1
        data = self.state["data"]
        training_config = self.experiment_config["experiment"]
        self.model_id = training_config["name"]
        if args.imbalanced_labels:
            if args.verbosity:
                print("Loading number of features by label for calculating class weights that will be applied on the loss function.")
            if "label_weights" in training_config:
                if args.verbosity:
                    print("Label weights already defined in the experiment config, ignoring feature distribution saved into state and using the config weights.")
            else:
                if args.verbosity:
                    print("Label weights not defined, calculating weights from training set feature distribution")
                num_features_by_label = data["training"]["num_features_by_label"]
                assert all(num_features > 0 for num_features in num_features_by_label.values())
                num_total_features = sum(num_features_by_label.values())
                training_config["label_weights"] = {
                    label: float(num_features / num_total_features)
                    for label, num_features in num_features_by_label.items()
                }
        if args.verbosity > 1:
            print("\nModel config is:")
            pprint.pprint(training_config)
            print()
        model = self.create_model(training_config)
        # Load training set consisting of pre-extracted features
        training_set, features_meta = system.load_features_as_dataset(
            # List of all .tfrecord files containing all training set samples
            list(data["training"]["features"].values()),
            training_config
        )
        if training_config.get("monitor_model_input", False):
            metrics_dir, training_set = model.enable_dataset_logger("training", training_set)
            self.make_named_dir(metrics_dir)
        # Same for the validation set
        validation_set, _ = system.load_features_as_dataset(
            list(data[training_config.get("validation_datagroup", "validation")]["features"].values()),
            training_config
        )
        if args.verbosity > 1:
            print("Compiling model")
        model.prepare(features_meta, training_config)
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoints = os.listdir(checkpoint_dir)
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, models.get_best_checkpoint(checkpoints, key="epoch"))
            if args.verbosity:
                print("Loading model weights from checkpoint file '{}'".format(checkpoint_path))
            model.load_weights(checkpoint_path)
        if args.verbosity:
            print("Starting training with model:\n")
            print(str(model))
            print()
        model.fit(training_set, validation_set, training_config)
        if args.verbosity:
            print("\nTraining finished\n")

    def run(self):
        super().run()
        return self.train()


class Evaluate(StatefulCommand):
    """Evaluate and predict using a trained model."""
    requires_state = State.has_model
    tasks = (
        "predict",
        "evaluate_test_set",
    )

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required = parser.add_argument_group("required args")
        required.add_argument("checkpoint_path",
            type=str,
            action=ExpandAbspath,
            help="Path to a checkpoint file that is used to load a trained model.")
        tasks = parser.add_argument_group("evaluation tasks")
        tasks.add_argument("--predict",
            type=str,
            metavar="AUDIOFILE_LIST",
            action=ExpandAbspath,
            help="Given a file containing a list of of audio files, predict a label for every audio file.")
        tasks.add_argument("--evaluate-test-set",
            action="store_true",
            help="Load test set paths from state and use evaluation metrics specified in config.")
        optional = parser.add_argument_group("evaluation options")
        optional.add_argument("--confusion-matrix-path",
            type=str,
            action=ExpandAbspath,
            help="Alternative, full path of the confusion matrix output.")
        optional.add_argument("--eval-result-dir",
            type=str,
            action=ExpandAbspath,
            help="Write evaluation results into this directory. If --confusion-matrix-path is also given, that path takes precedence over this directory.")
        return parser

    def get_eval_config(self):
        eval_config = self.experiment_config.copy()
        if "evaluation" in eval_config:
            for key, overrides in eval_config.pop("evaluation").items():
                if isinstance(overrides, dict):
                    eval_config[key].update(overrides)
                else:
                    eval_config[key] = overrides
        if "augmentation" in eval_config["features"]:
            del eval_config["features"]["augmentation"]
        if "repeat" in eval_config["experiment"]:
            del eval_config["experiment"]["repeat"]
        # Exhaust whole test set
        eval_config["experiment"]["validation_steps"] = None
        return eval_config

    def predict_paths(self, paths):
        args = self.args
        eval_config = self.get_eval_config()
        if args.verbosity > 1:
            print("Preparing model for prediction using evaluation config:")
            pprint.pprint(eval_config)
        if args.verbosity:
            print("Creating KerasWrapper '{}'".format(self.model_id))
        model = models.KerasWrapper(self.model_id, eval_config["experiment"]["model_definition"], device_str=eval_config["experiment"].get("tf_device_str"))
        # We assume the model has been trained on the training set features
        # We also assume all labels have features of same dimensions
        features_meta = system.load_features_meta(list(self.state["data"]["training"]["features"].values())[0])
        model.prepare(features_meta, eval_config["experiment"])
        model.load_weights(args.checkpoint_path)
        if args.verbosity:
            print("Extracting features from {} audio files".format(len(paths)))
        utterances = []
        extracted_paths = []
        print_progress = eval_config.get("print_progress", 0)
        extractor = transformations.files_to_features(paths, eval_config["features"])
        for i, (path, sequences) in enumerate(zip(paths, extractor), start=1):
            if len(sequences) > 0:
                utterances.append(sequences)
                extracted_paths.append(path)
            elif args.verbosity:
                print("Warning: skipping feature extraction for (possibly too short) file: '{}'".format(path))
            if print_progress and i % print_progress == 0:
                print(i, "files done")
        if print_progress:
            print("all files done")
        return zip(extracted_paths, model.predict(utterances))

    def predict(self):
        args = self.args
        if args.verbosity:
            print("Predicting labels for audio files listed in '{}'".format(args.predict))
        paths = [path for path, _ in system.load_audiofile_paths(args.predict)]
        index_to_label = collections.OrderedDict(
            sorted((i, label) for label, i in self.state["label_to_index"].items())
        )
        for path, prediction in self.predict_paths(paths):
            print("'{}':".format(path))
            print(("{:>8s}" * len(index_to_label)).format(*index_to_label.values()))
            amax = prediction.argmax()
            for i, p in enumerate(prediction):
                p_str = "{:8.3f}".format(p)
                if i == amax:
                    # This is the maximum likelihood, add ANSI code for bold text
                    p_str = "\x1b[1m" + p_str + "\x1b[0m"
                print(p_str, end='')
            print()

    def evaluate_test_set(self):
        args = self.args
        if args.verbosity:
            print("Evaluating test set")
        test_set = self.state["data"]["test"]
        label_to_index = self.state["label_to_index"]
        path_to_label = {p: label_to_index[l] for p, l in zip(test_set["paths"], test_set["labels"])}
        pred_labels = []
        true_labels = []
        for path, prediction in self.predict_paths(test_set["paths"]):
            pred_labels.append(prediction.argmax())
            true_labels.append(path_to_label[path])
        cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
        label_names = sorted(label_to_index.keys(), key=lambda l: label_to_index[l])
        fig, _ = visualization.draw_confusion_matrix(cm, label_names)
        figpath = os.path.join(self.eval_dir, "test-set-evaluation.svg")
        if args.confusion_matrix_path:
            figpath = args.confusion_matrix_path
        fig.savefig(figpath)

    def run(self):
        super().run()
        self.model_id = self.experiment_config["experiment"]["name"]
        args = self.args
        self.eval_dir = os.path.join(self.cache_dir, "evaluations")
        if args.eval_result_dir:
            self.eval_dir = args.eval_result_dir
        self.make_named_dir(self.eval_dir, "evaluation output")
        return self.run_tasks()


#TODO not usable yet
class Inspect(StatefulCommand):
    """Analysis of trained models."""
    requires_state = State.has_model
    tasks = (
        "count_model_params",
        "plot_progress",
    )

    #TODO needs create_model
    def count_model_params(self):
        training_config = self.experiment_config["experiment"]
        model = self.create_model(self.model_id, training_config)
        _, features_meta = system.load_features_as_dataset(
            list(self.state["data"]["training"]["features"].values()),
            training_config
        )
        model.prepare(features_meta, training_config)
        print(model_id, model.count_params())

    #TODO everything
    def plot_progress(self):
        args = self.args
        model_ids = sorted(self.model_ids, key=lambda k: (int(k.split("_")[1]), int(k.split("_")[3])))
        event_data = [
            {"model_id": model_id,
             "logdir": os.path.join(self.cache_dir, model_id, "tensorboard", "log")}
            for model_id in model_ids
        ]
        for data in event_data:
            logdir = data.pop("logdir")
            # Get most recent logdir, named by timestamp
            logdir = os.path.join(logdir, max(os.listdir(logdir)))
            # The metrics might have been saved under the 'validation' namespace, or all into the same event file
            if "validation" in os.listdir(logdir):
                metric = "epoch_loss"
                logdir = os.path.join(logdir, "validation")
            else:
                metric = "epoch_val_loss"
            # Get most recent event file
            event_file = os.path.join(logdir, system.get_most_recent_file(logdir))
            data.update({
                "event_file": event_file,
                "metric": metric,
            })
        if args.verbosity:
            print("Plotting metrics from {} event files".format(len(event_data)))
            if args.verbosity > 2:
                for data in event_data:
                    print(data["model_id"], data["event_file"])
                print()
        # Soften up names for legend
        for data in event_data:
            data["model_id"] = data["model_id"].split("_layer_")[-1].replace('_', ' ')
        fig, _ = visualization.draw_training_metrics_from_tf_events(
            event_data,
            xlabel="Epoch",
            ylabel="Loss",
            title=args.plot_title or args.plot_progress
        )
        figure_name = args.plot_progress + ".svg"
        figure_path = os.path.join(args.eval_result_dir or self.cache_dir, figure_name)
        fig.savefig(figure_path)
        if args.verbosity:
            print("Wrote metrics plot to '{}'".format(figure_path))


command_tree = [
    (Model, [Train, Evaluate]),
]
