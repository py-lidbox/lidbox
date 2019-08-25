import itertools
import os
import pprint
import sys

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
        config = self.experiment_config["features"]
        label_to_index = self.state["label_to_index"]
        for datagroup_name, datagroup in self.state["data"].items():
            if args.verbosity:
                print("Datagroup '{}' has {} audio files".format(datagroup_name, len(datagroup["paths"])))
            if "features" in datagroup:
                if args.verbosity:
                    print("Warning: datagroup '{}' already has features extracted, they will be overwritten:".format(datagroup_name))
                    features = datagroup["features"]
                    if type(features) is str:
                        print(feature)
                    else:
                        for f in datagroup["features"]:
                            pprint.pprint(f)
            datagroup["features"] = {}
            tfrecords_dir = os.path.join(args.cache_dir, datagroup_name)
            self.make_named_dir(tfrecords_dir)
            if args.verbosity:
                print("Extracting features by label")
            # The zip is very important here so as not to mess up the ordering of label-path pairs
            paths_sorted_by_label = sorted(zip(datagroup["labels"], datagroup["paths"]))
            # Extract all features by label, writing TFRecord files containing features for samples of only a single label
            for label, group in itertools.groupby(paths_sorted_by_label, key=lambda pair: pair[0]):
                labels, paths = list(zip(*group))
                assert all(l == label for l in labels), "Failed to group paths by label, expected all labels to be equal to '{}' but at least one was not".format(label)
                utterances = transformations.speech_dataset_to_utterances(
                    labels, paths,
                    utterance_length_ms=config["utterance_length_ms"],
                    utterance_offset_ms=config["utterance_offset_ms"],
                    apply_vad=config.get("apply_vad", False),
                    print_progress=config.get("print_progress", 0),
                    resample_to=config.get("resample_to")
                )
                features = transformations.utterances_to_features(
                    utterances,
                    label_to_index=label_to_index,
                    extractors=config["extractors"],
                    sequence_length=config["sequence_length"]
                )
                tfrecords_path = os.path.join(tfrecords_dir, label)
                wrote_path = system.write_features(features, tfrecords_path)
                datagroup["features"][label] = wrote_path
                if args.verbosity:
                    print("Wrote '{}' features to '{}'".format(datagroup_name, wrote_path))

    def count_features(self):
        args = self.args
        if args.verbosity:
            print("Counting all extracted features by datagroup and label")
        if not self.state_data_ok():
            return 1
        for datagroup_name, datagroup in self.state["data"].items():
            if "features" not in datagroup:
                print("Error: No features extracted for datagroup '{}', cannot count features".format(datagroup_name), file=sys.stderr)
                continue
            num_features_by_label = {}
            for label, features_file in datagroup["features"].items():
                dataset, _ = system.load_features_as_dataset([features_file])
                num_features_by_label[label] = int(dataset.reduce(0, lambda count, _: count + 1))
            datagroup["num_features_by_label"] = num_features_by_label
            if args.verbosity:
                print("Datagroup '{}' features count by label:".format(datagroup_name))
                for label, num_features in num_features_by_label.items():
                    print("  {}: {}".format(label, num_features))


    def run(self):
        super().run()
        return self.run_tasks()
