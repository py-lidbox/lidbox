"""
Command definitions for all tools.
"""
from datetime import datetime
import argparse
import collections
import itertools
import os
import pprint
import sys

import numpy as np

import speechbox
import speechbox.dataset as dataset
import speechbox.models as models
import speechbox.preprocess.features as features
import speechbox.preprocess.transformations as transformations
import speechbox.system as system


def create_argparser():
    parser = argparse.ArgumentParser(prog=speechbox.__name__, description=speechbox.__doc__)
    subparsers = parser.add_subparsers(
        title="tools",
        description="subcommands for different tasks",
    )
    # Create command line options for all valid commands
    for cmd in all_commands:
        subparser = cmd.create_argparser(subparsers)
        subparser.set_defaults(cmd_class=cmd)
    return parser


class ExpandAbspath(argparse.Action):
    """Simple argparse action to expand path arguments to full paths using os.path.abspath."""
    def __call__(self, parser, namespace, path, *args, **kwargs):
        setattr(namespace, self.dest, os.path.abspath(path))


class Command:
    """Base command with common helpers for all subcommands."""

    tasks = tuple()
    valid_datagroup_keys = ("paths", "labels", "checksums")

    @classmethod
    def create_argparser(cls, subparsers):
        parser = subparsers.add_parser(cls.__name__.lower(), description=cls.__doc__)
        parser.add_argument("cache_dir",
            type=str,
            action=ExpandAbspath,
            help="Speechbox cache for storing intermediate output such as extracted features.")
        parser.add_argument("experiment_config",
            type=str,
            action=ExpandAbspath,
            help="Path to a yaml-file containing the experiment configuration, e.g. hyperparameters, feature extractors etc.")
        parser.add_argument("--verbosity", "-v",
            action="count",
            default=0,
            help="Increases verbosity of output for each -v supplied (up to 4).")
        parser.add_argument("--run-cProfile",
            action="store_true",
            help="Do profiling on all commands and write results into a file in the working directory.")
        parser.add_argument("--src",
            type=str,
            action=ExpandAbspath,
            help="Source directory, depends on context.")
        parser.add_argument("--dst",
            type=str,
            action=ExpandAbspath,
            help="Target directory, depends on context.")
        parser.add_argument("--load-state",
            action="store_true",
            help="Load command state from the cache directory.")
        parser.add_argument("--save-state",
            action="store_true",
            help="Save command state to the cache directory.")
        return parser

    def __init__(self, args):
        self.args = args
        self.dataset_id = None
        self.experiment_config = {}
        self.state = {}

    def args_src_ok(self):
        args = self.args
        ok = True
        if not args.src:
            print("Error: Specify dataset source directory with --src.", file=sys.stderr)
            ok = False
        elif not os.path.isdir(args.src):
            print("Error: Source directory '{}' does not exist.".format(args.src), file=sys.stderr)
            ok = False
        return ok

    def args_dst_ok(self):
        args = self.args
        ok = True
        if not args.dst:
            print("Error: Specify dataset destination directory with --dst.", file=sys.stderr)
            ok = False
        else:
            self.make_named_dir(args.dst, "destination")
        return ok

    def state_data_ok(self):
        ok = True
        if not self.has_state() or "data" not in self.state:
            error_msg = (
                "Error: self.state does not have a 'data' key containing filepaths and labels, cannot extract features."
                " Either load an existing dataset definition from the cache with '--load-state' or create a new dataset split."
                "\nSee e.g. 'speechbox dataset --help'."
            )
            print(error_msg, file=sys.stderr)
            ok = False
        return ok

    def has_state(self):
        ok = True
        if not self.state:
            print("Error: No current state loaded, you need to use '--load-state' to load existing state from some cache directory.", file=sys.stderr)
            ok = False
        return ok

    def split_is_valid(self, original, split):
        """
        Check that all values from 'original' exist in 'split' and all values in 'split' are from 'original'.
        """
        ok = True
        train, val, test = split["training"], split["validation"], split["test"]
        for key in self.valid_datagroup_keys:
            val_original = set(original[key])
            val_splitted = set(train[key]) | set(val[key]) | set(test[key])
            if val_original != val_splitted:
                ok = False
                error_msg = "Error: key '{}' has an invalid split, some information was probably lost during the split.".format(key)
                error_msg += "\nValues in the original data but not in the split:\n"
                error_msg += ' '.join(str(o) for o in val_original - val_splitted)
                error_msg += "\nValues in the split but not in the original data:\n"
                error_msg += ' '.join(str(s) for s in val_splitted - val_original)
                print(error_msg, file=sys.stderr)
        return ok

    def make_named_dir(self, path, name=None):
        if not os.path.isdir(path):
            if self.args.verbosity:
                if name:
                    print("Creating {} directory '{}'".format(name, path))
                else:
                    print("Creating directory '{}'".format(path))
            os.makedirs(path)

    def load_state(self):
        args = self.args
        state_path = os.path.join(args.cache_dir, "speechbox_state.json.gz")
        if args.verbosity:
            print("Loading state from '{}'".format(state_path))
        self.state = system.load_gzip_json(state_path)

    def save_state(self):
        args = self.args
        self.make_named_dir(args.cache_dir, "cache")
        state_path = os.path.join(args.cache_dir, "speechbox_state.json.gz")
        if args.verbosity:
            print("Saving state to '{}'".format(state_path))
        system.dump_gzip_json(self.state, state_path)

    def merge_datagroups_from_state(self):
        data = self.state["data"]
        return {
            key: list(itertools.chain(*(datagroup[key] for datagroup in data.values())))
            for key in self.valid_datagroup_keys
        }

    def run(self):
        args = self.args
        if args.verbosity > 1:
            print("Running tool '{}' with arguments:".format(self.__class__.__name__.lower()))
            pprint.pprint(vars(args))
            print()
        if args.verbosity:
            print("Loading experiment config from '{}'".format(args.experiment_config))
        self.experiment_config = system.load_yaml(args.experiment_config)
        if args.verbosity > 1:
            print("Experiment config is:")
            pprint.pprint(self.experiment_config)
            print()
        if "src" in self.experiment_config:
            if args.src and args.verbosity:
                print("Dataset source directory given in experiment config yaml as '{}', the directory given with --src: '{}' will be ignored.".format(self.experiment_config["src"], args.src))
            args.src = self.experiment_config["src"]
        self.dataset_id = self.experiment_config["dataset_id"]
        if args.load_state:
            self.load_state()
        if args.verbosity > 1:
            print("Running with initial state:")
            pprint.pprint(self.state, depth=3)
            print()

    def run_tasks(self):
        given_tasks = [getattr(self, task_name) for task_name in self.__class__.tasks if getattr(self.args, task_name)]
        if not given_tasks:
            print("Error: No tasks given, doing nothing", file=sys.stderr)
            return 2
        for task in given_tasks:
            ret = task()
            if ret:
                return ret

    def exit(self):
        if self.args.save_state:
            self.save_state()


class Dataset(Command):
    """Dataset analysis and manipulation."""

    tasks = (
        "walk",
        "parse",
        "split",
        "check_split",
        "check_integrity",
        "to_kaldi",
        "compare_state",
        "augment",
    )

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--walk",
            action="store_true",
            help="Walk over a dataset with a dataset walker, gathering audio file paths and labels. By default, all paths and labels are written into experiment state under key 'all'.")
        parser.add_argument("--output",
            type=str,
            action=ExpandAbspath,
            help="Write all (MD5-checksum, wavpath, label) pairs returned by the walk to this file. Columns are separated by a single space.")
        parser.add_argument("--parse",
            action="store_true",
            help="TODO")
        parser.add_argument("--resampling-rate",
            type=int,
            help="If given with --parse, all wavfile output will be resampled to this sampling rate.")
        parser.add_argument("--check",
            action="store_true",
            help="Check that every file in the walk is unique and contains audio. Might take a long time since every file will be opened briefly with sox and/or librosa.")
        parser.add_argument("--split",
            type=str,
            choices=dataset.all_split_types,
            help="Create a random training-validation-test split for a dataset and store the splits as datagroups.")
        parser.add_argument("--check-split",
            type=str,
            choices=dataset.all_split_types,
            help="Check that all datagroups are disjoint by the given split type.")
        parser.add_argument("--check-integrity",
            action="store_true",
            help="Recompute MD5 checksum values for every file and compare the new values to the existing, cached values.")
        parser.add_argument("--compare-state",
            type=str,
            action=ExpandAbspath,
            help="Compare state in this directory to the current state.")
        parser.add_argument("--to-kaldi",
            type=str,
            action=ExpandAbspath,
            help="Write dataset paths and current state as valid input for the Kaldi toolkit into the given directory. Creates wav.scp and utt2spk.")
        parser.add_argument("--augment",
            action="store_true",
            help="Apply augmentation on all paths in dataset and write output into directory given by '--dst'. Augmentation config is defined in the experiment config under sox_transform.")
        return parser

    def walk(self):
        args = self.args
        if args.verbosity:
            print("Walking over dataset '{}'".format(self.dataset_id))
        if self.state:
            if args.verbosity:
                print("Using paths from state, gathered during some previous walk")
            walker_config = self.merge_datagroups_from_state()
        else:
            if args.verbosity:
                print("Gathering paths from directory '{}'.".format(args.src))
            if not self.args_src_ok():
                return 1
            walker_config = {
                "dataset_root": args.src,
            }
        walker_config["sample_frequency"] = args.resampling_rate
        dataset_walker = dataset.get_dataset_walker(self.dataset_id, walker_config)
        if args.check:
            if args.verbosity:
                print("'--check' given, invalid files will be ignored during the walk")
            dataset_iter = dataset_walker.walk(check_read=True, check_duplicates=True, verbosity=args.verbosity)
        else:
            if args.verbosity:
                print("'--check' not given, walk will be much faster but invalid files will not be ignored")
            dataset_iter = dataset_walker.walk(verbosity=args.verbosity)
        if args.output:
            if args.verbosity:
                print("Will write all (checksum, wavfile, label) pairs to '{}'".format(args.output))
            with open(args.output, 'w') as out_f:
                for label, wavpath, checksum in dataset_iter:
                    print(checksum, wavpath, label, sep=' ', file=out_f)
        else:
            data = self.state.get("data", {})
            if args.verbosity and "all" in data:
                print("Warning: overwriting existing paths in self.state for dataset '{}'".format(self.dataset_id))
            # Gather all
            labels, paths, checksums = [], [], []
            for label, path, checksum in dataset_iter:
                labels.append(label)
                paths.append(path)
                checksums.append(system.md5sum(path))
            walk_data = {
                "label_to_index": dataset_walker.make_label_to_index_dict(),
                "labels": labels,
                "paths": paths,
                "checksums": checksums,
            }
            # Merge to state under key 'all'
            self.state["data"] = dict(data, all=walk_data)
            self.state["source_directory"] = args.src

    def check_split(self):
        args = self.args
        if args.verbosity:
            print("Checking that all datagroups are disjoint by split type '{}'".format(args.check_split))
        if not self.state_data_ok():
            return 1
        datagroups = self.state["data"]
        if len(datagroups) == 1:
            print("Error: Only one datagroup found: '{}', cannot check that it is disjoint with other datagroups since other datagroups do not exist".format(list(datagroups.keys())[0]))
            return 1
        if args.check_split == "by-file":
            for a, b in itertools.combinations(datagroups.keys(), r=2):
                print("'{}' vs '{}' ... ".format(a, b), flush=True, end='')
                # Group all filepaths by MD5 checksums of file contents
                duplicates = collections.defaultdict(list)
                a_paths = zip(datagroups[a]["checksums"], datagroups[a]["paths"])
                b_paths = zip(datagroups[b]["checksums"], datagroups[b]["paths"])
                for checksum, path in itertools.chain(a_paths, b_paths):
                    duplicates[checksum].append(path)
                # Filter out all non-singleton groups
                duplicates = [paths for paths in duplicates.values() if len(paths) > 1]
                if duplicates:
                    error_msg = "error: datagroups are not disjoint"
                    if args.verbosity > 2:
                        error_msg += ", following files have equal checksums:"
                        print(error_msg)
                        for paths in duplicates:
                            for path in paths:
                                print(path)
                    else:
                        print(error_msg)
                else:
                    print("ok")
        elif args.check_split == "by-speaker":
            for a, b in itertools.combinations(datagroups.keys(), r=2):
                print("'{}' vs '{}' ... ".format(a, b), flush=True, end='')
                # Group all filepaths by speaker ids parsed from filepaths of the given dataset
                parse_speaker_id = dataset.get_dataset_walker_cls(self.dataset_id).parse_speaker_id
                a_speakers = set(parse_speaker_id(path) for path in datagroups[a]["paths"])
                b_speakers = set(parse_speaker_id(path) for path in datagroups[b]["paths"])
                shared_speakers = a_speakers & b_speakers
                if shared_speakers:
                    error_msg = "error: datagroups are not disjoint"
                    if args.verbosity > 2:
                        error_msg += ", following speakers have samples in both datagroups:"
                        print(error_msg)
                        for s in shared_speakers:
                            print(s)
                    else:
                        print(error_msg)
                else:
                    print("ok")
        else:
            print("Error: Unknown split type '{}', cannot check".format(args.check_split), file=sys.stderr)
            return 1

    def parse(self):
        args = self.args
        if args.verbosity:
            print("Parsing dataset '{}'".format(self.dataset_id))
        if not (self.args_src_ok() and self.args_dst_ok()):
            return 1
        parser_config = {
            "dataset_root": args.src,
            "output_dir": args.dst,
            "resampling_rate": args.sample_frequency,
        }
        parser = dataset.get_dataset_parser(self.dataset_id, parser_config)
        num_parsed = 0
        if not args.verbosity:
            for _ in parser.parse():
                num_parsed += 1
        else:
            for output in parser.parse():
                num_parsed += 1
                if any(output):
                    status, out, err = output
                    msg = "Warning:"
                    if status:
                        msg += " exit code: {}".format(status)
                    if out:
                        msg += " stdout: '{}'".format(out)
                    if err:
                        msg += " stderr: '{}'".format(err)
                    print(msg)
        if args.verbosity:
            print(num_parsed, "files processed")

    def split(self):
        args = self.args
        if args.verbosity:
            print("Creating a training-validation-test split for dataset '{}' using split type '{}'".format(self.dataset_id, args.split))
        datagroups = self.state.get("data", {})
        if "all" in datagroups:
            if args.verbosity:
                print("Using existing dataset paths from current state, possibly loaded from the cache")
                if args.src:
                    print("Ignoring dataset source directory '{}'".format(args.src))
            walker_config = self.merge_datagroups_from_state()
        elif len(datagroups) > 0:
            error_msg = (
                "Error: The loaded state already has a split into {} different datagroups:\n".format(len(datagroups)) +
                '\n'.join(str(datagroup) for datagroup in datagroups)
            )
            print(error_msg, file=sys.stderr)
            return 1
        else:
            if args.verbosity:
                print("Dataset paths not found in self.state['data'], walking over whole dataset starting at '{}' to gather paths".format(args.src))
            if not self.args_src_ok():
                return 1
            walker_config = {
                "dataset_root": args.src,
            }
        dataset_walker = dataset.get_dataset_walker(self.dataset_id, walker_config)
        # The label-to-index mapping is common for all datagroups in the split
        self.state["label_to_index"] = dataset_walker.make_label_to_index_dict()
        if args.split == "by-speaker":
            splitter = transformations.dataset_split_samples_by_speaker
        else:
            splitter = transformations.dataset_split_samples
        training_set, validation_set, test_set = splitter(dataset_walker, verbosity=args.verbosity)
        split = {
            "training": {
                "paths": training_set[0],
                "labels": training_set[1],
                "checksums": training_set[2],
            },
            "validation": {
                "paths": validation_set[0],
                "labels": validation_set[1],
                "checksums": validation_set[2],
            },
            "test": {
                "paths": test_set[0],
                "labels": test_set[1],
                "checksums": test_set[2],
            }
        }
        # Merge split into self.state["data"], such that keys in split take priority and overwrite existing keys
        self.state["data"] = dict(self.state.get("data", {}), **split)
        if "all" in self.state["data"]:
            # Sanity check that we did not lose any information during the split
            if not self.split_is_valid(self.state["data"]["all"], split):
                return 1
            del self.state["data"]["all"]

    def to_kaldi(self):
        args = self.args
        kaldi_dir = args.to_kaldi
        self.make_named_dir(kaldi_dir, "kaldi output")
        for datagroup_name, datagroup in self.state["data"].items():
            paths, labels, checksums = datagroup["paths"], datagroup["labels"], datagroup["checksums"]
            if args.verbosity:
                print("'{}', containing {} paths".format(datagroup_name, len(paths)))
            dataset_walker = dataset.get_dataset_walker(self.dataset_id, {"paths": paths, "labels": labels, "checksums": checksums})
            if not hasattr(dataset_walker, "parse_speaker_id"):
                print("Error: Dataset walker '{}' does not support parsing speaker ids from audio file paths, cannot create 'utt2spk'. Define a parse_speaker_id for the walker class.".format(str(dataset_walker)), file=sys.stderr)
                return 1
            output_dir = os.path.join(kaldi_dir, datagroup_name)
            self.make_named_dir(output_dir)
            wav_scp, utt2spk = [], []
            for _, wavpath, _ in dataset_walker.walk(verbosity=args.verbosity):
                file_id = str(dataset_walker.get_file_id(wavpath))
                speaker_id = str(dataset_walker.parse_speaker_id(wavpath))
                wav_scp.append((file_id, wavpath))
                # Add speaker- prefix to make sure no sorting script tries to automatically switch to numerical sort
                utt2spk.append((file_id, "speaker-" + speaker_id))
            wav_scp.sort()
            utt2spk.sort()
            with open(os.path.join(output_dir, "wav.scp"), 'w') as wav_scp_f:
                for line in wav_scp:
                    print(*line, file=wav_scp_f)
            with open(os.path.join(output_dir, "utt2spk"), 'w') as utt2spk_f:
                for line in utt2spk:
                    print(*line, file=utt2spk_f)
            with open(os.path.join(output_dir, "mfcc.conf"), 'w') as mfcc_conf:
                print("--use-energy=false", file=mfcc_conf)
                print("--sample-frequency={:d}".format(dataset_walker.sample_frequency or 16000), file=mfcc_conf)

    def compare_state(self):
        args = self.args
        other_path = os.path.join(args.compare_state, "state.json.gz")
        if args.verbosity:
            print("Loading other state from '{}'".format(other_path))
        other_state = system.load_gzip_json(other_path)
        if not self.has_state():
            return 1
        if args.verbosity > 1:
            print("Other state is:")
            pprint.pprint(other_state, depth=3)
            print()
        duplicates = collections.defaultdict(list)
        for datagroup in itertools.chain(other_state["data"].values(), self.state["data"].values()):
            for checksum, path in zip(datagroup["checksums"], datagroup["paths"]):
                duplicates[checksum].append(path)
        # Filter out all non-singleton groups
        duplicates = [paths for paths in duplicates.values() if len(paths) > 1]
        if duplicates:
            print("These files have exactly equal contents:")
            for paths in duplicates:
                for p in paths:
                    print(p)
                print()
        else:
            print("Did not find any files with exactly equal contents")

    def check_integrity(self):
        if not self.has_state():
            return 1
        args = self.args
        if args.verbosity:
            print("Checking integrity of all files by recomputing MD5 checksums.")
        mismatches = collections.defaultdict(list)
        for datagroup_name, datagroup in self.state["data"].items():
            for checksum, path in zip(datagroup["checksums"], datagroup["paths"]):
                new_checksum = system.md5sum(path)
                if checksum != new_checksum:
                    mismatches[datagroup_name].append((checksum, new_checksum, path))
        num_mismatching = sum(len(m) for m in mismatches.values())
        print("Found {} files with mismatching MD5 checksums.".format(num_mismatching))
        for datagroup_name, mismatch_list in mismatches.items():
            print("Datagroup '{}', {} files:".format(datagroup_name, len(mismatch_list)))
            print("old", "new", "path")
            for old, new, path in mismatch_list:
                print("{} {} {}".format(old, new, path))

    def augment(self):
        if not self.state_data_ok():
            return 1
        if "source_directory" not in self.state:
            print("Error: no source directory in cache, run --walk and --save-state on some dataset in directory specified by --src to gather all paths into the cache.", file=sys.stderr)
            return 1
        args = self.args
        if not args.dst:
            if args.verbosity:
                print("No augmentation destination given, assuming cache directory '{}'".format(args.cache_dir))
            dst = os.path.join(args.cache_dir, "augmented-data")
        else:
            dst = args.dst
        prefix = self.state["source_directory"]
        if args.verbosity:
            print("Augmenting dataset from '{}' into '{}'".format(prefix, dst))
        augment_config = self.experiment_config["augmentation"]
        if "list" in augment_config:
            augment_config = augment_config["list"]
        elif "cartesian_product" in augment_config:
            all_kwargs = augment_config["cartesian_product"].items()
            flattened_kwargs = [[(aug_type, v) for v in aug_values] for aug_type, aug_values in all_kwargs]
            augment_config = [dict(kwargs) for kwargs in itertools.product(*flattened_kwargs)]
        if args.verbosity > 1:
            print("Full config for augmentation:")
            pprint.pprint(augment_config)
            print()
        print_progress = self.experiment_config.get("print_progress", 0)
        src_paths_by_datagroup = {
            datagroup_key: datagroup["paths"]
            for datagroup_key, datagroup in self.state["data"].items()
        }
        dst_paths_by_datagroup = {datagroup_key: [] for datagroup_key in src_paths_by_datagroup}
        num_augmented = 0
        for aug_kwargs in augment_config:
            if args.verbosity:
                print("Augmenting by:")
                for aug_type, aug_value in aug_kwargs.items():
                    print(aug_type, aug_value)
                print()
            for datagroup_key, src_paths in src_paths_by_datagroup.items():
                # Create sox src-dst file pairs for each transformation
                dst_paths = []
                for src_path in src_paths:
                    # Use directory structure from the source dir but replace the prefix
                    dst_path = src_path.replace(prefix, dst)
                    dirname, basename = os.path.split(dst_path)
                    augdir = '__'.join((str(aug_type) + '-' + str(aug_value)) for aug_type, aug_value in aug_kwargs.items())
                    target_dir = os.path.join(dirname, augdir)
                    # Make dirs if they do not exist
                    self.make_named_dir(target_dir)
                    dst_paths.append(os.path.join(target_dir, basename))
                for src_path, dst_path in system.apply_sox_transformer(src_paths, dst_paths, **aug_kwargs):
                    if dst_path is None:
                        if args.verbosity:
                            print("Warning, sox failed to transform '{}' from '{}'".format(dst_path, src_path))
                    else:
                        num_augmented += 1
                        if args.verbosity > 3:
                            print("augmented {} to {}".format(src_path, dst_path))
                    # We still want to gather all None's to know which failed
                    dst_paths_by_datagroup[datagroup_key].append(dst_path)
                    if print_progress > 0 and num_augmented % print_progress == 0:
                        print(num_augmented, "files augmented")

    def run(self):
        super().run()
        return self.run_tasks()


class Preprocess(Command):
    """Feature extraction."""

    tasks = ("extract_features",)

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--extract-features",
            action="store_true",
            help="Perform feature extraction on whole dataset.")
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

    def run(self):
        super().run()
        return self.run_tasks()


class Model(Command):
    """Model training and evaluation."""

    tasks = ("train", "evaluate_test_set", "predict")

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--no-save-model",
            action="store_true",
            help="Do not save model state, such as epoch checkpoints, into the cache directory.")
        parser.add_argument("--model-id",
            type=str,
            help="Use this value as the model name instead of the one in the experiment yaml-file.")
        parser.add_argument("--train",
            action="store_true",
            help="Run model training using configuration from the experiment yaml.  Checkpoints are written at every epoch into the cache dir, unless --no-save-model was given.")
        parser.add_argument("--evaluate-test-set",
            action="store_true",
            help="Evaluate model on test set")
        parser.add_argument("--predict",
            type=str,
            action=ExpandAbspath,
            help="Predict labels for all audio files listed in the given file, one per line.")
        return parser

    def create_model(self, model_id, model_config):
        args = self.args
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_cache_dir = os.path.join(args.cache_dir, model_id)
        tensorboard_dir = os.path.join(model_cache_dir, "tensorboard", "log", now_str)
        default_tensorboard_config = {
            "log_dir": tensorboard_dir,
            "write_graph": False,
        }
        tensorboard_config = dict(default_tensorboard_config, **model_config.get("tensorboard", {}))
        checkpoint_dir = os.path.join(model_cache_dir, "checkpoints")
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
        checkpoints_config = dict(default_checkpoints_config, **model_config.get("checkpoints", {}))
        callbacks_kwargs = {
            "checkpoints": None if args.no_save_model else checkpoints_config,
            "early_stopping": model_config.get("early_stopping"),
            "tensorboard": tensorboard_config,
        }
        if not args.train:
            if args.verbosity > 1:
                print("Not training, will not use keras callbacks")
            callbacks_kwargs = {"device_str": model_config.get("eval_device")}
        else:
            self.make_named_dir(tensorboard_dir, "tensorboard")
            if not args.no_save_model:
                self.make_named_dir(checkpoint_dir, "checkpoints")
        if args.verbosity > 1:
            print("KerasWrapper callback parameters will be set to:")
            pprint.pprint(callbacks_kwargs)
            print()
        return models.KerasWrapper(model_id, **callbacks_kwargs)

    @staticmethod
    def get_loss_as_float(checkpoint_filename):
        return float(checkpoint_filename.split("loss")[-1].split(".hdf5")[0])

    def load_best_weights(self):
        args = self.args
        checkpoints_dir = os.path.join(args.cache_dir, self.model_id, "checkpoints")
        all_checkpoints = os.listdir(checkpoints_dir)
        if not all_checkpoints:
            print("Error: Cannot load model weights since there are no keras checkpoints in '{}'".format(checkpoints_dir), file=sys.stderr)
            return 1
        best_checkpoint = os.path.join(checkpoints_dir, min(all_checkpoints, key=self.get_loss_as_float))
        if args.verbosity:
            print("Loading weights from keras checkpoint '{}'".format(best_checkpoint))
        self.state["model"].load_weights(best_checkpoint)

    def train(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for training")
        if not self.state_data_ok():
            return 1
        data = self.state["data"]
        model_config = self.experiment_config["model"]
        if args.verbosity > 1:
            print("\nModel config is:")
            pprint.pprint(model_config)
            print()
        model = self.state["model"]
        # Load training set consisting of pre-extracted features
        training_set, features_meta = system.load_features_as_dataset(
            # List of all .tfrecord files containing all training set samples
            [data["training"]["features"]],
            model_config
        )
        # Same for the validation set
        validation_set, _ = system.load_features_as_dataset(
            [data["validation"]["features"]],
            model_config
        )
        model.prepare(features_meta, model_config)
        if args.verbosity:
            print("\nStarting training with model:\n")
            print(str(model))
            print()
        model.fit(training_set, validation_set, model_config)
        if args.verbosity:
            print("\nTraining finished\n")

    def evaluate_test_set(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for evaluation")
        model = self.state["model"]
        model_config = self.experiment_config["model"]
        if not self.has_state() or "test" not in self.state["data"]:
            print("Error: test set paths not found", file=sys.stderr)
            return 1
        test_set, features_meta = system.load_features_as_dataset(
            [self.state["data"]["test"]["features"]],
            model_config
        )
        model.prepare(features_meta, model_config)
        self.load_best_weights()
        model.evaluate(test_set, model_config)

    def predict(self):
        args = self.args
        if args.verbosity:
            print("Predicting labels for audio files listed in '{}'".format(args.predict))
        config = self.experiment_config
        if args.verbosity > 1:
            print("Preparing model for prediction")
        model_config = config["model"]
        model = self.state["model"]
        features_meta = system.load_features_meta(self.state["data"]["training"]["features"])
        model.prepare(features_meta, model_config)
        self.load_best_weights()
        paths = list(system.load_audiofile_paths(args.predict))
        if args.verbosity:
            print("Extracting features from {} audio files".format(len(paths)))
        utterances = []
        # Split to utterances and features by file
        for path in paths:
            utterance_chunks = transformations.speech_dataset_to_utterances(
                [0], [path],
                utterance_length_ms=config["utterance_length_ms"],
                utterance_offset_ms=config["utterance_offset_ms"],
                apply_vad=config.get("apply_vad", False),
                print_progress=config.get("print_progress", 0)
            )
            features = transformations.utterances_to_features(
                utterance_chunks,
                label_to_index=[0],
                extractors=config["extractors"],
                sequence_length=config["sequence_length"]
            )
            # Evaluate features generator while dropping dummy labels
            features = [feat for feat, dummy_label in features]
            if features:
                utterances.append((path, features))
            elif args.verbosity:
                print("Unable to extract features for (possibly too short) sample: '{}'".format(path))
        index_to_label = {i: label for label, i in self.state["label_to_index"].items()}
        for path, features in utterances:
            features = np.array(features)
            prediction = model.predict(features)
            sequence_means = prediction.mean(axis=0)
            label_index = sequence_means.argmax()
            label, prob = index_to_label[label_index], sequence_means[label_index]
            print("'{}' with probability {:.3f}, '{}'".format(label, prob, os.path.basename(path)))


    def run(self):
        super().run()
        args = self.args
        model_config = self.experiment_config["model"]
        self.model_id = args.model_id if args.model_id else model_config["name"]
        if args.verbosity:
            print("Creating KerasWrapper '{}'".format(self.model_id))
        self.state["model"] = self.create_model(self.model_id, model_config)
        return self.run_tasks()


all_commands = (
    Dataset,
    Preprocess,
    Model,
)
