import functools
import itertools
import multiprocessing
import os
import pprint
import sys

from speechbox.commands.base import State, Command, StatefulCommand
import speechbox.system as system
import speechbox.preprocess.transformations as transformations


def extract_features_from_task(task):
    """
    Feature extractor function with a serializable parameter object for parallel and/or async execution.
    """
    (label, group), (tfrecords_dir, config, label_to_index) = task
    labels = []
    paths = []
    for l, p in group:
        labels.append(l)
        paths.append(p)
    assert all(l == label for l in labels), "Failed to group paths by label, expected all labels to be equal to '{}' but at least one was not".format(label)
    utterances = transformations.speech_dataset_to_utterances(
        labels, paths,
        utterance_length_ms=config["utterance_length_ms"],
        utterance_offset_ms=config["utterance_offset_ms"],
        apply_vad=config.get("apply_vad", False),
        print_progress=config.get("print_progress", 0),
        slide_over_all=config.get("slide_over_all", True),
    )
    features = transformations.utterances_to_features(
        utterances,
        label_to_index=label_to_index,
        extractors=config["extractors"],
        sequence_length=config["sequence_length"]
    )
    return label, system.write_features(features, os.path.join(tfrecords_dir, label))


class Features(Command):
    """Feature extraction and feature analysis."""
    pass


class Extract(StatefulCommand):
    """Extract features from audio files using parameters from the config yaml."""
    requires_state = State.has_paths

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        optional = parser.add_argument_group("extract options")
        optional.add_argument("--num-workers",
            type=int,
            default=1,
            help="How many parallel processes to use when extracting features. Defaults to 1, which means multiprocessing will not be used.")
        return parser

    def extract(self):
        args = self.args
        if args.verbosity:
            print("Starting feature extraction")
        if not self.state_data_ok():
            return 1
        config = self.experiment_config["features"]
        if args.verbosity > 1:
            print("Using feature extraction parameters:")
            pprint.pprint(config)
            print()
        for datagroup_name in config["datagroups"]:
            datagroup = self.state["data"][datagroup_name]
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
            tfrecords_dir = os.path.join(self.cache_dir, datagroup_name)
            self.make_named_dir(tfrecords_dir)
            if args.verbosity:
                print("Extracting '{}' features by label using {} parallel workers".format(datagroup_name, args.num_workers))
            # The zip is very important here so as not to mess up the ordering of label-path pairs
            paths_sorted_by_label = sorted(zip(datagroup["labels"], datagroup["paths"]))
            # Group all paths by label
            label_groups = itertools.groupby(paths_sorted_by_label, key=lambda pair: pair[0])
            # We must evaluate the group generator since the key function is not picklable
            label_groups = [(label, list(group)) for label, group in label_groups]
            task_args = (tfrecords_dir, config, self.state["label_to_index"])
            tasks = ((group, task_args) for group in label_groups)
            if args.num_workers > 1:
                with multiprocessing.Pool(args.num_workers) as pool:
                    res = list(pool.imap_unordered(extract_features_from_task, tasks))
            else:
                res = [extract_features_from_task(t) for t in tasks]
            for label, wrote_path in res:
                datagroup["features"][label] = wrote_path
                if args.verbosity:
                    print("Wrote '{}' features for label '{}' into '{}'".format(datagroup_name, label, wrote_path))

    def run(self):
        super().run()
        return self.extract()


class Count(StatefulCommand):
    """Count amount of extracted features by label."""
    requires_state = State.has_features

    def count(self):
        args = self.args
        if args.verbosity:
            print("Counting all extracted features by datagroup and label")
        if not self.state_data_ok():
            return 1
        for datagroup_name, datagroup in self.state["data"].items():
            if "features" not in datagroup:
                print("Error: No features extracted for datagroup '{}', cannot count features".format(datagroup_name), file=sys.stderr)
                continue
            labels, feature_files = list(zip(*datagroup["features"].items()))
            num_features_by_label = dict(system.count_all_features_parallel(labels, feature_files))
            datagroup["num_features_by_label"] = num_features_by_label
            if args.verbosity:
                print("Datagroup '{}' features count by label:".format(datagroup_name))
                for label, num_features in num_features_by_label.items():
                    print("  {}: {}".format(label, num_features))

    def run(self):
        super().run()
        return self.count()


command_tree = [
    (Features, [Extract, Count]),
]
