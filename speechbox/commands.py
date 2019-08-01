import argparse
import pprint
import os

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


class Command:
    """Base command that only prints its arguments to stdout."""

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
        return parser

    def __init__(self, args):
        self.args = args
        self.state = {}

    def check_src_dst(self):
        ok = True
        if not self.args.src:
            print("Error: Specify dataset source directory with --src.")
            ok = False
        elif not os.path.isdir(self.args.src):
            print("Error: Source directory '{}' does not exist.".format(self.args.src))
            ok = False
        if not self.args.dst:
            print("Error: Specify dataset destination directory with --dst.")
            ok = False
        elif not os.path.isdir(self.args.dst):
            if self.args.create_dirs:
                if self.args.verbosity:
                    print("Creating destination directory '{}'".format(self.args.dst))
                os.makedirs(self.args.dst)
            else:
                print("Error: Destination directory '{}' does not exist.".format(self.args.dst))
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
            help="Walk over a dataset, printing labels and absolute paths for each sample, one per line.")
        parser.add_argument("--parse",
            action="store_true",
            help="Parse a dataset according to parameters set in the config file, given as '--config-file'.")
        parser.add_argument("--src",
            type=str,
            help="Source directory, depends on context.")
        parser.add_argument("--dst",
            type=str,
            help="Target directory, depends on context.")
        return parser

    def walk(self):
        if self.args.verbosity:
            print("Walking over dataset '{}'".format(self.args.dataset_key))

    def parse(self):
        if self.args.verbosity:
            print("Parsing dataset '{}'".format(self.args.dataset_key))
        if not self.check_src_dst():
            return 1
        parser_config = {
            "dataset_root": self.args.src,
            "output_dir": self.args.dst,
        }
        parser = datasets.get_dataset_parser(self.args.dataset_key, parser_config)
        if not self.args.verbosity:
            for _ in parser.parse():
                pass
        else:
            for output in parser.parse():
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

    def run(self):
        super().run()
        if self.args.walk:
            self.walk()
        if self.args.parse:
            self.parse()


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
