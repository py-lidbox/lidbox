import argparse
import itertools
import os
import pprint
import sys

import speechbox.system as system


class ExpandAbspath(argparse.Action):
    """Simple argparse action to expand path arguments to full paths using os.path.abspath."""
    def __call__(self, parser, namespace, path, *args, **kwargs):
        setattr(namespace, self.dest, os.path.abspath(path))


class State:
    none = "no-state"
    has_paths = "has-paths"
    has_features = "has-features"
    has_model = "has-model"


class Command:
    """Stateless command base class."""

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
            help="Do profiling on all Python function calls and write results into a file in the working directory.")
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
            os.makedirs(path)

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

    def exit(self):
        pass


class StatefulCommand(Command):
    """Base command with state that can be loaded and saved between runs."""

    tasks = tuple()
    requires_state = State.none

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required_args = parser.add_argument_group("state definition", description="Required arguments for defining state updates.")
        required_args.add_argument("experiment_config",
            type=str,
            action=ExpandAbspath,
            help="Path to a yaml-file containing the experiment configuration, e.g. path to the cache directory, feature extractors, model hyperparameters etc.")
        suboptions = parser.add_argument_group("state interaction")
        suboptions.add_argument("--save-state",
            action="store_true",
            help="Save command state to the cache directory.")
        return parser

    def __init__(self, args):
        super().__init__(args)
        self.dataset_id = None
        self.cache_dir = None
        self.experiment_config = {}
        self.state = {}

    def state_data_ok(self):
        ok = True
        if "data" not in self.state:
            error_msg = (
                "Error: self.state does not have a 'data' key containing filepaths and labels."
                "\nSee e.g. 'speechbox dataset --help'."
            )
            print(error_msg, file=sys.stderr)
            ok = False
        return ok

    def state_ok(self):
        ok = True
        if self.state["state"] != self.requires_state:
            print("Error: command '{}' has incorrect state '{}' when '{}' was required".format(self), self.state["state"], self.requires_state, file=sys.stderr)
            ok = False
        return ok

    def load_state(self):
        args = self.args
        state_path = os.path.join(self.cache_dir, "speechbox_state.json.gz")
        if args.verbosity:
            print("Loading state from '{}'".format(state_path))
        self.state = system.load_gzip_json(state_path)

    def save_state(self):
        args = self.args
        self.make_named_dir(self.cache_dir, "cache")
        state_path = os.path.join(self.cache_dir, "speechbox_state.json.gz")
        if args.verbosity:
            print("Saving state to '{}'".format(state_path))
        system.dump_gzip_json(self.state, state_path)

    def run(self):
        super().run()
        args = self.args
        if args.verbosity > 1:
            print("Running subcommand '{}' with arguments:".format(self.__class__.__name__.lower()))
            pprint.pprint(vars(args))
            print()
        if args.verbosity:
            print("Loading experiment config from '{}'".format(args.experiment_config))
        self.experiment_config = system.load_yaml(args.experiment_config)
        if args.verbosity > 1:
            print("Experiment config is:")
            pprint.pprint(self.experiment_config)
            print()
        self.cache_dir = os.path.abspath(self.experiment_config["cache"])
        if args.verbosity > 1:
            print("Cache dir is '{}'".format(self.cache_dir))
        self.dataset_id = self.experiment_config["dataset"]["key"]
        if self.requires_state != State.none:
            self.load_state()
        if args.verbosity > 1:
            print("Running with initial state:")
            pprint.pprint(self.state, depth=None if args.debug else 3)
            print()

    def exit(self):
        if self.args.save_state:
            self.save_state()
        super().exit()
