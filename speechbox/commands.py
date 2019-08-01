"""
Command definitions for all tools.
"""
import argparse
import os
import pprint
import sys

import yaml

import speechbox
import speechbox.datasets as datasets


def create_argparser():
    parser = argparse.ArgumentParser(prog=speechbox.__name__)
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
        parser.add_argument("--create-dirs",
            action="store_true",
            help="Create non-existing directories when needed.")
        parser.add_argument("--src",
            type=str,
            action=ExpandAbspath,
            help="Source directory, depends on context.")
        parser.add_argument("--dst",
            type=str,
            action=ExpandAbspath,
            help="Target directory, depends on context.")
        return parser

    def __init__(self, args):
        self.args = args
        self.state = {}

    def check_src(self):
        ok = True
        if not self.args.src:
            print("Error: Specify dataset source directory with --src.", file=sys.stderr)
            ok = False
        elif not os.path.isdir(self.args.src):
            print("Error: Source directory '{}' does not exist.".format(self.args.src), file=sys.stderr)
            ok = False
        return ok

    def check_dst(self):
        ok = True
        if not self.args.dst:
            print("Error: Specify dataset destination directory with --dst.", file=sys.stderr)
            ok = False
        elif not os.path.isdir(self.args.dst):
            if self.args.create_dirs:
                if self.args.verbosity:
                    print("Creating destination directory '{}'".format(self.args.dst))
                os.makedirs(self.args.dst)
            else:
                print("Error: Destination directory '{}' does not exist.".format(self.args.dst), file=sys.stderr)
                ok = False
        return ok

    def run(self):
        if self.args.verbosity > 1:
            print("Running tool '{}' with arguments:".format(self.__class__.__name__.lower()))
            pprint.pprint(vars(self.args))
            print()
        if self.args.config_file:
            if self.args.verbosity:
                print("Parsing config file '{}'".format(self.args.config_file))
            with open(self.args.config_file) as f:
                self.state["config"] = yaml.safe_load(f)
            if self.args.verbosity > 1:
                print("Config file contents:")
                pprint.pprint(self.state["config"])
                print()


class Dataset(Command):
    """Dataset analysis and manipulation."""

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("dataset_key",
            choices=datasets.all_datasets,
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
            help="Walk over a dataset checking every file. Might take a long time. Implies verbosity=2 during walk.")
        parser.add_argument("--split",
            choices=datasets.all_split_types,
            help="Create a random training-validation-test split for a dataset into --dst.")
        return parser

    def walk(self):
        if self.args.verbosity:
            print("Walking over dataset '{}'".format(self.args.dataset_key))
        if self.args.dataset_key == "unittest":
            # Special case, there is a mini-subset of the Mozilla Common Voice dataset in the source tree of this package
            self.args.src = speechbox._get_unittest_data_dir()
        if not self.check_src():
            return 1
        walker_config = {
            "dataset_root": self.args.src,
            "sampling_rate_override": self.args.resampling_rate,
        }
        dataset_walker = datasets.get_dataset_walker(self.args.dataset_key, walker_config)
        for label, wavpath in dataset_walker.walk(verbosity=self.args.verbosity):
            print(wavpath, label)

    def check(self):
        if self.args.verbosity:
            print("Checking integrity of dataset '{}'".format(self.args.dataset_key))
        if self.args.dataset_key == "unittest":
            self.args.src = speechbox._get_unittest_data_dir()
        if not self.check_src():
            return 1
        walker_config = {
            "dataset_root": self.args.src,
        }
        dataset_walker = datasets.get_dataset_walker(self.args.dataset_key, walker_config)
        for _ in dataset_walker.walk(check_duplicates=True, check_read=True, verbosity=2):
            pass

    def parse(self):
        if self.args.verbosity:
            print("Parsing dataset '{}'".format(self.args.dataset_key))
        if not (self.check_src() and self.check_dst()):
            return 1
        parser_config = {
            "dataset_root": self.args.src,
            "output_dir": self.args.dst,
            "resampling_rate": self.args.resampling_rate,
        }
        parser = datasets.get_dataset_parser(self.args.dataset_key, parser_config)
        num_parsed = 0
        if not self.args.verbosity:
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
        if self.args.verbosity:
            print(num_parsed, "files processed")

    def split(self):
        if self.args.verbosity:
            print("Creating a training-validation-test split for dataset '{}'".format(self.args.dataset_key))
        if not (self.check_src() and self.check_dst()):
            return 1

    def run(self):
        super().run()
        if self.args.walk:
            ret = self.walk()
            if ret: return ret
        if self.args.parse:
            ret = self.parse()
            if ret: return ret
        if self.args.check:
            ret = self.check()
            if ret: return ret
        if self.args.split:
            ret = self.split()
            if ret: return ret


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
