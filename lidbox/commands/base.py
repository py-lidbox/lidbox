import argparse
import itertools
import os
import sys

import lidbox.system as system
from lidbox import yaml_pprint


class ExpandAbspath(argparse.Action):
    """Simple argparse action to expand path arguments to full paths using os.path.abspath."""
    def __call__(self, parser, namespace, path, *args, **kwargs):
        setattr(namespace, self.dest, os.path.abspath(path))


class BaseCommand:

    tasks = tuple()

    @classmethod
    def create_argparser(cls, subparsers):
        parser = subparsers.add_parser(
            cls.__name__.lower(),
            description=cls.__doc__,
            add_help=False
        )
        group = parser.add_argument_group("global options", description="Optional, global arguments for all subcommands and tasks.")
        group.add_argument("--help", "-h",
            action="help",
            help="Show this message and exit.")
        group.add_argument("--verbosity", "-v",
            action="count",
            default=0,
            help="Increases output verbosity for each -v supplied up to 4 (-vvvv).")
        group.add_argument("--debug",
            action="store_true",
            default=False,
            help="Set maximum verbosity and enable extra debugging information regardless of performance impacts.")
        group.add_argument("--run-cProfile",
            action="store_true",
            default=False,
            help="Do profiling on all Python function calls and write results into a file in the working directory.")
        group.add_argument("--run-tf-profiler",
            action="store_true",
            default=False,
            help="Run the TensorFlow profiler and create a trace event file.")
        return parser

    def __init__(self, args):
        self.args = args

    def make_named_dir(self, path, name=None):
        if not os.path.isdir(path):
            if self.args.verbosity:
                if name:
                    print("Creating {} directory '{}'".format(name, path))
                else:
                    print("Creating directory '{}'".format(path))
        os.makedirs(path, exist_ok=True)

    def run_tasks(self):
        given_tasks = [
            getattr(self, task_name)
            for task_name in self.__class__.tasks
            if getattr(self.args, task_name)
        ]
        if not given_tasks:
            print("Error: No tasks given, doing nothing", file=sys.stderr)
            return 2
        for task in given_tasks:
            # The user might pipe output of some command to e.g. UNIX 'head' or 'tail',
            # which may send a SIGPIPE back to tell us there is no need to print more data.
            # Python throws a BrokenPipeError when it receives that signal so let's just exit with code 0 in that case.
            try:
                ret = task()
            except BrokenPipeError:
                return 0
            if ret:
                return ret

    def run(self):
        if self.args.debug:
            print("\n\nWarning: running in debug mode\n\n")
            self.args.verbosity = 4


class Command(BaseCommand):

    tasks = tuple()

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required_args = parser.add_argument_group("required", description="Required arguments")
        required_args.add_argument("experiment_config",
            type=str,
            action=ExpandAbspath,
            help="Path to a yaml-file containing the experiment configuration, e.g. path to the cache directory, feature extractors, model hyperparameters etc.")
        return parser

    def __init__(self, args):
        super().__init__(args)
        self.dataset_id = None
        self.cache_dir = None
        self.experiment_config = {}

    def run(self):
        super().run()
        args = self.args
        if args.verbosity > 1:
            print("Running subcommand '{}' with arguments:".format(self.__class__.__name__.lower()))
            yaml_pprint(vars(args))
            print()
        if args.verbosity:
            print("Loading experiment config from '{}'".format(args.experiment_config))
        self.experiment_config = system.load_yaml(args.experiment_config)
        if args.verbosity > 1:
            print("Experiment config is:")
            yaml_pprint(self.experiment_config)
            print()
        self.cache_dir = os.path.abspath(self.experiment_config["cache"])
        if args.verbosity > 1:
            print("Cache dir is '{}'".format(self.cache_dir))
        self.dataset_id = self.experiment_config["dataset"]["key"]
