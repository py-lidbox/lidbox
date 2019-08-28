from datetime import datetime
import collections
import os
import pprint
import shutil
import sys

from speechbox.commands import ExpandAbspath
from speechbox.commands.base import StatefulCommand
import speechbox.dataset as dataset
import speechbox.models as models
import speechbox.preprocess.transformations as transformations
import speechbox.system as system
import speechbox.visualization as visualization


class Model(StatefulCommand):
    """Model training and evaluation."""

    tasks = ("train", "evaluate_test_set", "predict", "plot_progress", "count_model_params")

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--no-save-model",
            action="store_true",
            help="Do not save model state, such as epoch checkpoints, into the cache directory.")
        parser.add_argument("--model-id",
            action="append",
            help="Use this value as the model name instead of the one in the experiment yaml-file. Can be specified multiple times for the tasks that support it. If a task does not support multiple model ids, only the first one is used and rest are ignored.")
        parser.add_argument("--train",
            action="store_true",
            help="Run model training using configuration from the experiment yaml.  Checkpoints are written at every epoch into the cache dir, unless --no-save-model was given.")
        parser.add_argument("--imbalanced-labels",
            action="store_true",
            help="Apply weighting on imbalanced labels during training by using a pre-calculated feature distribution.")
        parser.add_argument("--evaluate-test-set",
            choices=("loss", "confusion-matrix"),
            action="append",
            help="Evaluate model on test set")
        parser.add_argument("--group-by-speaker",
            action="store_true",
            help="Concatenate all audio files from the same speaker using the order they appear in the list of test set paths.")
        parser.add_argument("--confusion-matrix-path",
            type=str,
            action=ExpandAbspath,
            help="Alternative, full path of the confusion matrix output.")
        parser.add_argument("--predict",
            type=str,
            action=ExpandAbspath,
            help="Predict labels for all audio files listed in the given file, one per line.")
        parser.add_argument("--reset-tensorboard",
            action="store_true",
            help="Delete tensorboard directory from previous runs for this model.")
        parser.add_argument("--count-model-params",
            action="store_true",
            help="Count total amount of parameters in all layers of given models.")
        parser.add_argument("--reset-checkpoints",
            action="store_true",
            help="Delete checkpoints from previous runs for this model.")
        parser.add_argument("--eval-result-dir",
            type=str,
            action=ExpandAbspath,
            help="Write evaluation results into this directory. If --confusion-matrix-path is also given, that path takes precedence over this directory.")
        parser.add_argument("--plot-progress",
            type=str,
            help="Load metrics from the most recent tensorboard event file and write progress plots with this figure name (not path).")
        parser.add_argument("--plot-title", type=str)
        return parser

    #FIXME
    def count_model_params(self):
        training_config = self.experiment_config["experiment"]
        for model_id in self.model_ids:
            model = self.create_model(model_id, training_config)
            _, features_meta = system.load_features_as_dataset(
                list(self.state["data"]["training"]["features"].values()),
                training_config
            )
            model.prepare(features_meta, training_config)
            print(model_id, model.count_params())

    def create_model(self, model_id, config):
        args = self.args
        model_cache_dir = os.path.join(args.cache_dir, model_id)
        tensorboard_log_dir = os.path.join(model_cache_dir, "tensorboard", "log")
        if args.reset_tensorboard and os.path.isdir(tensorboard_log_dir):
            if args.verbosity:
                print("Clearing tensorboard directory '{}'".format(tensorboard_log_dir))
            shutil.rmtree(tensorboard_log_dir)
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard_dir = os.path.join(tensorboard_log_dir, now_str)
        default_tensorboard_config = {
            "log_dir": tensorboard_dir,
            "write_graph": True,
            "update_freq": "epoch",
        }
        if args.debug:
            default_tensorboard_config.update({
                "update_freq": "batch",
                "histogram_freq": 1,
                "embeddings_freq": 1,
            })
        tensorboard_config = dict(default_tensorboard_config, **config.get("tensorboard", {}))
        checkpoint_dir = os.path.join(model_cache_dir, "checkpoints")
        if args.reset_checkpoints and os.path.isdir(checkpoint_dir):
            if args.verbosity:
                print("Clearing checkpoint directory '{}'".format(checkpoint_dir))
            shutil.rmtree(checkpoint_dir)
        checkpoint_format = "epoch{epoch:02d}_loss{val_loss:.2f}.hdf5"
        default_checkpoints_config = {
            "filepath": os.path.join(checkpoint_dir, checkpoint_format),
            "load_weights_on_restart": True,
            # Save models only when the validation loss is minimum, i.e. 'best'
            "mode": "min",
            "monitor": "val_loss",
            "save_best_only": True,
            "verbose": 0,
        }
        checkpoints_config = dict(default_checkpoints_config, **config.get("checkpoints", {}))
        callbacks_kwargs = {
            "checkpoints": None if args.no_save_model else checkpoints_config,
            "early_stopping": config.get("early_stopping"),
            "tensorboard": tensorboard_config,
        }
        if not args.train:
            if args.verbosity > 1:
                print("Not training, will not use keras callbacks")
            callbacks_kwargs = {"device_str": config.get("tf_device_str")}
        else:
            self.make_named_dir(tensorboard_dir, "tensorboard")
            if not args.no_save_model:
                self.make_named_dir(checkpoint_dir, "checkpoints")
        if args.verbosity > 1:
            print("KerasWrapper callback parameters will be set to:")
            pprint.pprint(callbacks_kwargs)
            print()
        return models.KerasWrapper(model_id, config["model_definition"], **callbacks_kwargs)

    def get_eval_config(self):
        eval_config = self.experiment_config.copy()
        if "evaluation"in eval_config:
            if self.args.verbosity:
                print("Additional parameters specified in the config to be used during evaluation, they will take precedence over existing config values used during training")
            for key, overrides in eval_config.pop("evaluation").items():
                eval_config[key].update(overrides)
        if "augmentation" in eval_config["features"]:
            del eval_config["features"]["augmentation"]
        if "repeat" in eval_config["experiment"]:
            del eval_config["experiment"]["repeat"]
        # Exhaust whole test set
        eval_config["experiment"]["validation_steps"] = None
        return eval_config

    @staticmethod
    def get_loss_as_float(checkpoint_filename):
        return float(checkpoint_filename.split("loss")[-1].split(".hdf5")[0])

    def get_best_weights_checkpoint(self):
        args = self.args
        checkpoints_dir = os.path.join(args.cache_dir, self.model_ids[0], "checkpoints")
        all_checkpoints = os.listdir(checkpoints_dir)
        if not all_checkpoints:
            print("Error: Cannot load model weights since there are no keras checkpoints in '{}'".format(checkpoints_dir), file=sys.stderr)
            return 1
        best_checkpoint = os.path.join(checkpoints_dir, min(all_checkpoints, key=self.get_loss_as_float))
        if args.verbosity:
            print("Loading weights from keras checkpoint '{}'".format(best_checkpoint))
        return best_checkpoint

    def train(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for training")
        if not self.state_data_ok():
            return 1
        data = self.state["data"]
        training_config = self.experiment_config["experiment"]
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
        model_id = self.model_ids[0]
        model = self.create_model(model_id, training_config)
        # Load training set consisting of pre-extracted features
        training_set, features_meta = system.load_features_as_dataset(
            # List of all .tfrecord files containing all training set samples
            list(data["training"]["features"].values()),
            training_config
        )
        if args.debug:
            metrics_dir, training_set = model.enable_dataset_logger("training", training_set)
            self.make_named_dir(metrics_dir)
        # Same for the validation set
        validation_set, _ = system.load_features_as_dataset(
            list(data["validation"]["features"].values()),
            training_config
        )
        model.prepare(features_meta, training_config)
        if args.verbosity:
            print("Starting training with model:\n")
            print(str(model))
            print()
        model.fit(training_set, validation_set, training_config)
        if args.verbosity:
            print("\nTraining finished\n")

    #FIXME a horrible mess
    def evaluate_test_set(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for evaluation")
        eval_config = self.get_eval_config()
        if args.verbosity > 1:
            print("Initializing model and extracting features for evaluation with config:")
            pprint.pprint(eval_config)
            print()
        experiment_config = eval_config["experiment"]
        model = self.create_model(model_id, experiment_config)
        if not self.state_data_ok():
            return 1
        if "test" not in self.state["data"]:
            print("Error: test set paths not found", file=sys.stderr)
            return 1
        test_set_data = self.state["data"]["test"]
        if args.verbosity > 1:
            print("Test set has {} paths".format(len(test_set_data["paths"])))
        features_meta = system.load_features_meta(
            list(self.state["data"]["training"]["features"].values())[0]
        )
        assert features_meta["sequence_length"] == eval_config["features"]["sequence_length"], "Mismatching sequence lengths"
        model.prepare(features_meta, experiment_config)
        best_checkpoint = self.get_best_weights_checkpoint()
        model.load_weights(best_checkpoint)
        test_set_paths = test_set_data["paths"]
        test_set_labels = test_set_data["labels"]
        if args.group_by_speaker:
            if args.verbosity:
                print("Grouping test set paths by speaker for concatenation")
            output_dir = os.path.join(args.cache_dir, "test", "wav")
            self.make_named_dir(output_dir)
            # Keep labels paired up with the paths
            test_set = zip(test_set_paths, test_set_labels)
            test_set_paths = []
            test_set_labels = []
            for speaker, samples in dataset.group_samples_by_speaker(test_set, self.dataset_id):
                if args.verbosity > 3:
                    print("Speaker '{}' has {} files".format(speaker, len(samples)))
                assert all(label == samples[0][1] for _, label in samples), "Expected all speakers to have samples of one language only."
                concat_name = speaker + '_'.join(os.path.basename(path).split(".wav")[0].replace(speaker, '') for path, _ in samples)
                concat_wavpath = os.path.join(output_dir, concat_name + ".wav")
                if os.path.exists(concat_wavpath):
                    if args.verbosity > 1:
                        print("Speaker '{}' already has file '{}'".format(speaker, concat_wavpath))
                else:
                    concat_wav = system.concatenate_wavs([system.read_wavfile(w) for w, _ in samples])
                    system.write_wav(concat_wavpath, concat_wav)
                test_set_paths.append(concat_wavpath)
                test_set_labels.append(samples[0][1])
            if args.verbosity > 3:
                print("Grouped all files by speaker:")
                for path, label in zip(test_set_paths, test_set_labels):
                    print(path, label)
                print()
        # Convert string labels to one hot encoding
        label_to_index = self.state["label_to_index"]
        # Extract utterances
        if args.verbosity > 1:
            print("Extracting features for all files in the test set with config:")
            pprint.pprint(eval_config["features"])
            print()
        transformer = transformations.files_to_features(
                test_set_paths[:5],
                test_set_labels[:5],
            eval_config["features"],
            label_to_index
        )
        test_set = list(transformer)
        if args.verbosity > 3:
            print("Extracted features by file:")
            for path, utterances, target in test_set:
                print(path, utterances.shape, target.argmax())
        for evaluation_type in args.evaluate_test_set:
            if evaluation_type == "loss":
                index_to_label = collections.OrderedDict(
                    sorted((i, label) for label, i in label_to_index.items())
                )
                model.predict(test_set, index_to_label)
                if False:
                    metrics = [float(m) for m in metrics]
                    named_metrics = dict(zip(model.model.metrics_names, metrics))
                    if "eval_result" not in self.state:
                        self.state["eval_result"] = {}
                    self.state["eval_result"].update({
                        "loss_function": experiment_config["loss"],
                        "metrics": named_metrics,
                        "model": best_checkpoint,
                        "num_model_params": model.count_params()
                    })
            elif evaluation_type == "confusion-matrix":
                cm = model.evaluate_confusion_matrix(
                    [u for _, u, _ in test_set],
                    [t for _, _, t in test_set],
                )
                # Sort labels by index for labeling plot axes
                label_names = sorted(label_to_index, key=lambda label: label_to_index[label])
                fig, _ = visualization.draw_confusion_matrix(cm, label_names)
                if args.confusion_matrix_path:
                    cm_figure_path = args.confusion_matrix_path
                else:
                    figure_name = "confusion-matrix_test-set_model-{}.svg".format(os.path.basename(best_checkpoint))
                    cm_figure_path = os.path.join(args.eval_result_dir or args.cache_dir, figure_name)
                if "eval_result" not in self.state:
                    self.state["eval_result"] = {}
                self.state["eval_result"]["confusion_matrix"] = cm_figure_path
                self.state["eval_result"]["_cm_fig"] = fig
            else:
                print("Error: unknown test set evaluation type '{}'".format(evaluation_type))

    def predict(self):
        args = self.args
        if args.verbosity:
            print("Predicting labels for audio files listed in '{}'".format(args.predict))
        if args.verbosity > 1:
            print("Preparing model for prediction")
        eval_config = self.get_eval_config()
        model_id = self.model_ids[0]
        if args.verbosity:
            print("Creating KerasWrapper '{}'".format(model_id))
            if args.verbosity > 1:
                print("Using config:")
                pprint.pprint(eval_config)
                print()
        model = self.create_model(model_id, eval_config["experiment"])
        features_meta = system.load_features_meta(self.state["data"]["training"]["features"])
        model.prepare(features_meta, eval_config["experiment"])
        model.load_weights(self.get_best_weights_checkpoint())
        paths = [path for path, _ in system.load_audiofile_paths(args.predict)]
        if args.verbosity:
            print("Extracting features from {} audio files".format(len(paths)))
        utterances = []
        extracted_paths = []
        for path, utterance in transformations.files_to_utterances(paths, config):
            if utterance is None:
                if args.verbosity > 1:
                    print("Warning: could not extract features from (possibly too short) file '{}'".format(path))
            else:
                utterances.append(utterance)
                extracted_paths.append(path)
        index_to_label = collections.OrderedDict(
            sorted((i, label) for label, i in self.state["label_to_index"].items())
        )
        for path, prediction in zip(extracted_paths, model.predict(utterances)):
            print("'{}':".format(path))
            print(("{:>8s}" * len(index_to_label)).format(*index_to_label.values()))
            for p in prediction:
                print("{:8.3f}".format(p), end='')
            print()

    #FIXME
    def plot_progress(self):
        args = self.args
        model_ids = sorted(self.model_ids, key=lambda k: (int(k.split("_")[1]), int(k.split("_")[3])))
        event_data = [
            {"model_id": model_id,
             "logdir": os.path.join(args.cache_dir, model_id, "tensorboard", "log")}
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
        figure_path = os.path.join(args.eval_result_dir or args.cache_dir, figure_name)
        fig.savefig(figure_path)
        if args.verbosity:
            print("Wrote metrics plot to '{}'".format(figure_path))

    def run(self):
        super().run()
        args = self.args
        experiment_name = self.experiment_config["experiment"]["name"]
        self.model_ids = args.model_id if args.model_id else [experiment_name]
        return self.run_tasks()

    def exit(self):
        args = self.args
        if args.eval_result_dir:
            self.make_named_dir(args.eval_result_dir, "evaluation results")
        if "eval_result" in self.state:
            results = self.state["eval_result"]
            if "confusion_matrix" in results:
                results.pop("_cm_fig").savefig(results["confusion_matrix"])
                if args.verbosity:
                    print("Wrote confusion matrix to '{}'".format(results["confusion_matrix"]))
            if args.eval_result_dir:
                self.make_named_dir(args.eval_result_dir, "evaluation results")
                eval_result_path = os.path.join(args.eval_result_dir, "evaluation.json")
                if args.verbosity:
                    print("Writing evaluation results to '{}'".format(eval_result_path))
                    if args.verbosity > 1:
                        pprint.pprint(results)
                system.append_json(results, eval_result_path)
        super().exit()
