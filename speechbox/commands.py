"""
Command definitions for all tools.
"""
import argparse
import collections
import itertools
import json
import os
import pprint
import sys

import yaml

import speechbox
import speechbox.dataset as dataset
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

    @classmethod
    def create_argparser(cls, subparsers):
        parser = subparsers.add_parser(cls.__name__.lower(), description=cls.__doc__)
        parser.add_argument("--config-file",
            type=str,
            help="Path to the speechbox configuration yaml-file.")
        parser.add_argument("--verbosity", "-v",
            action="count",
            default=0,
            help="Increase verbosity of output to stdout.")
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
        parser.add_argument("--cache-dir",
            type=str,
            action=ExpandAbspath,
            help="Use this directory as a cache to store intermediate output and program state.")
        parser.add_argument("--load-state",
            action="store_true",
            help="Load command state from the cache directory.")
        parser.add_argument("--save-state",
            action="store_true",
            help="Save command state to the cache directory.")
        return parser

    def __init__(self, args):
        self.args = args
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
        elif not os.path.isdir(args.dst):
            if args.verbosity:
                print("Creating destination directory '{}'".format(args.dst))
            os.makedirs(args.dst)
        return ok

    def load_state(self):
        args = self.args
        state_json = os.path.join(args.cache_dir, "state.json")
        if args.verbosity:
            print("Loading state from '{}'".format(state_json))
        with open(state_json) as f:
            self.state = json.load(f)

    def save_state(self):
        args = self.args
        if not os.path.isdir(args.cache_dir):
            if args.verbosity:
                print("Creating cache directory '{}'".format(args.cache_dir))
            os.makedirs(args.cache_dir)
        state_json = os.path.join(args.cache_dir, "state.json")
        if args.verbosity:
            print("Saving state to '{}'".format(state_json))
        with open(state_json, "w") as f:
            json.dump(self.state, f)

    def run(self):
        args = self.args
        if args.verbosity > 1:
            print("Running tool '{}' with arguments:".format(self.__class__.__name__.lower()))
            pprint.pprint(vars(args))
            print()
        if args.cache_dir and not (args.load_state or args.save_state):
            print("Warning: neither --load-state nor --save-state was given, --cache-dir does nothing in this case.", file=sys.stderr)
        if args.load_state:
            self.load_state()
        if args.config_file:
            if args.verbosity:
                print("Parsing config file '{}'".format(args.config_file))
            with open(args.config_file) as f:
                self.state["config"] = yaml.safe_load(f)
            if args.verbosity > 1:
                print("Config file contents:")
                pprint.pprint(self.state["config"])
                print()

    def exit(self):
        if self.args.save_state:
            self.save_state()


class Dataset(Command):
    """Dataset analysis and manipulation."""

    tasks = ("walk", "parse", "split", "check")

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("dataset_id",
            choices=dataset.all_datasets,
            help="Which dataset to use.")
        parser.add_argument("--walk",
            action="store_true",
            help="Walk over a dataset, printing wavpath-label pairs.")
        parser.add_argument("--parse",
            action="store_true",
            help="Parse a dataset according to parameters set in the config file, given as '--config-file'.")
        parser.add_argument("--resampling-rate",
            type=int,
            help="If given with --parse, all wavfile output will be resampled to this sampling rate.")
        parser.add_argument("--check",
            action="store_true",
            help="Walk over a dataset checking every file. Might take a long time since every file will be opened.")
        parser.add_argument("--split",
            choices=dataset.all_split_types,
            help="Create a random training-validation-test split for a dataset into --dst.")
        return parser

    def walk(self):
        args = self.args
        if args.verbosity:
            print("Walking over dataset '{}'".format(args.dataset_id))
        if not args_src_ok():
            return 1
        walker_config = {
            "dataset_root": args.src,
            "sampling_rate_override": args.resampling_rate,
        }
        dataset_walker = dataset.get_dataset_walker(args.dataset_id, walker_config)
        for label, wavpath in dataset_walker.walk(verbosity=args.verbosity):
            print(wavpath, label)

    def check(self):
        args = self.args
        if args.verbosity:
            print("Checking integrity of dataset '{}'".format(args.dataset_id))
        if "split" in self.state:
            if args.verbosity:
                print("Dataset split defined in self.state, checking all files by split")
                print("Checking that the dataset splits are disjoint by file contents")
            split = self.state["split"]
            for a, b in itertools.combinations(split.keys(), r=2):
                print("'{}' vs '{}' ... ".format(a, b), flush=True, end='')
                # Group all filepaths by hashes on the file contents
                duplicates = collections.defaultdict(list)
                for path in itertools.chain(split[a]["paths"], split[b]["paths"]):
                    duplicates[system.md5sum(path)].append(path)
                # Filter out all singleton groups
                duplicates = [paths for paths in duplicates.values() if len(paths) > 1]
                if duplicates:
                    print("error: datasets not disjoint, following files have equal content hashes:")
                    for paths in duplicates:
                        for path in paths:
                            print(path)
                else:
                    print("ok")
            if args.verbosity:
                print("Checking all audio files in the dataset")
            dataset_walker = dataset.get_dataset_walker(args.dataset_id)
            for split_name, split in self.state["split"].items():
                paths, labels = split["paths"], split["labels"]
                if args.verbosity:
                    print("'{}', containing {} paths and {} labels, of which {} labels are unique".format(split_name, len(paths), len(labels), len(set(labels))))
                dataset_walker.overwrite_target_paths(paths, labels)
                for _ in dataset_walker.walk(check_duplicates=True, check_read=True, verbosity=args.verbosity):
                    pass
        else:
            if args.verbosity:
                print("Dataset split not defined in self.state, checking dataset from its root directory '{}'".format(args.src))
            if not args_src_ok():
                return 1
            walker_config = {
                "dataset_root": args.src,
            }
            dataset_walker = dataset.get_dataset_walker(args.dataset_id, walker_config)
            for _ in dataset_walker.walk(check_duplicates=True, check_read=True, verbosity=args.verbosity):
                pass

    def parse(self):
        args = self.args
        if args.verbosity:
            print("Parsing dataset '{}'".format(args.dataset_id))
        if not (args_src_ok() and args_dst_ok()):
            return 1
        parser_config = {
            "dataset_root": args.src,
            "output_dir": args.dst,
            "resampling_rate": args.resampling_rate,
        }
        parser = dataset.get_dataset_parser(args.dataset_id, parser_config)
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
            print("Creating a training-validation-test split for dataset '{}' using split type '{}'".format(args.dataset_id, args.split))
        if not args_src_ok():
            return 1
        walker_config = {
            "dataset_root": args.src,
        }
        dataset_walker = dataset.get_dataset_walker(args.dataset_id, walker_config)
        if args.split == "by-speaker":
            splitter = transformations.dataset_split_samples_by_speaker
        else:
            splitter = transformations.dataset_split_samples
        training_set, validation_set, test_set = splitter(dataset_walker)
        self.state["split"] = {
            "training": {
                "paths": training_set[0],
                "labels": training_set[1]
            },
            "validation": {
                "paths": validation_set[0],
                "labels": validation_set[1]
            },
            "test": {
                "paths": test_set[0],
                "labels": test_set[1]
            }
        }

    def run(self):
        super().run()
        args = self.args
        if args.dataset_id == "unittest" and not args.src:
            # Special case, there is a mini-subset of the Mozilla Common Voice dataset in the source tree of this package
            args.src = speechbox._get_unittest_data_dir()
        for attr in self.__class__.tasks:
            if getattr(args, attr):
                ret = getattr(self, attr)()
                if ret:
                    return ret



class Preprocess(Command):
    """Feature extraction."""

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("-p")
        return parser

    def run(self):
        super().run()


class Train(Command):
    """Model training."""

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("-t")
        return parser

    def run(self):
        super().run()


class Evaluate(Command):
    """Prediction and evaluation using trained models."""

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("-e")
        return parser

    def run(self):
        super().run()


all_commands = (
    Dataset,
    Preprocess,
    Train,
    Evaluate,
)
