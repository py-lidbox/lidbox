import itertools
import os
import pprint

from speechbox.commands.base import StatefulCommand
import speechbox.system as system
import speechbox.preprocess.transformations as transformations


class Preprocess(StatefulCommand):
    """Feature extraction and feature analysis."""

    tasks = ("extract_features", "count_features")

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--extract-features",
            action="store_true",
            help="Perform feature extraction on whole dataset.")
        parser.add_argument("--count-features",
            action="store_true",
            help="Count the amount of extracted features in every datagroup.")
        return parser

    def extract_features(self):
        args = self.args
        if args.verbosity:
            print("Starting feature extraction")
        if not self.state_data_ok():
            return 1
        config = self.experiment_config
        label_to_index = self.state["label_to_index"]
        for datagroup_name, datagroup in self.state["data"].items():
            if args.verbosity:
                print("Datagroup '{}' has {} audio files".format(datagroup_name, len(datagroup["paths"])))
            labels, paths = datagroup["labels"], datagroup["paths"]
            utterances = transformations.speech_dataset_to_utterances(
                labels, paths,
                utterance_length_ms=config["utterance_length_ms"],
                utterance_offset_ms=config["utterance_offset_ms"],
                apply_vad=config.get("apply_vad", False),
                print_progress=config.get("print_progress", 0)
            )
            features = transformations.utterances_to_features(
                utterances,
                label_to_index=label_to_index,
                extractors=config["extractors"],
                sequence_length=config["sequence_length"]
            )
            target_path = os.path.join(args.cache_dir, datagroup_name)
            wrote_path = system.write_features(features, target_path)
            datagroup["features"] = wrote_path
            if args.verbosity:
                print("Wrote '{}' features to '{}'".format(datagroup_name, wrote_path))

    def count_features(self):
        args = self.args
        if args.verbosity:
            print("Counting all extracted features by datagroup")
        if not self.state_data_ok():
            return 1
        for datagroup_name, datagroup in self.state["data"].items():
            if "features" not in datagroup:
                print("Error: No features extracted for datagroup '{}', cannot count features".format(datagroup_name), file=sys.stderr)
                continue
            num_features = system.count_dataset(datagroup["features"])
            print("'{}' has {} features".format(datagroup_name, num_features))

    def run(self):
        super().run()
        return self.run_tasks()
