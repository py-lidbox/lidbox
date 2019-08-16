import itertools
import os
import pprint
import sys

from speechbox.commands import ExpandAbspath
import speechbox.system as system


class Command:
    """Stateless, minimal command skeleton."""

    tasks = tuple()

    @classmethod
    def create_argparser(cls, subparsers):
        parser = subparsers.add_parser(cls.__name__.lower(), description=cls.__doc__)
        parser.add_argument("--verbosity", "-v",
            action="count",
            default=0,
            help="Increases verbosity of output for each -v supplied (up to 4).")
        parser.add_argument("--run-cProfile",
            action="store_true",
            help="Do profiling on all commands and write results into a file in the working directory.")
        return parser

    def __init__(self, args):
        self.args = args

    def args_src_ok(self):
        args = self.args
        ok = True
        if not args.src:
            print("Error: Specify dataset source directory", file=sys.stderr)
            ok = False
        elif not os.path.isdir(args.src):
            print("Error: Source directory '{}' does not exist.".format(args.src), file=sys.stderr)
            ok = False
        return ok

    def args_dst_ok(self):
        args = self.args
        ok = True
        if not args.dst:
            #TODO self.error that prints usage
            print("Error: Specify dataset destination directory with", file=sys.stderr)
            ok = False
        else:
            self.make_named_dir(args.dst, "destination")
        return ok

    def make_named_dir(self, path, name=None):
        if not os.path.isdir(path):
            if self.args.verbosity:
                if name:
                    print("Creating {} directory '{}'".format(name, path))
                else:
                    print("Creating directory '{}'".format(path))
            os.makedirs(path)

    def run_tasks(self):
        given_tasks = [getattr(self, task_name) for task_name in self.__class__.tasks if getattr(self.args, task_name)]
        if not given_tasks:
            print("Error: No tasks given, doing nothing", file=sys.stderr)
            return 2
        for task in given_tasks:
            ret = task()
            if ret:
                return ret

    def run(self):
        pass



class StatefulCommand(Command):
    """Base command with state that can be loaded and saved between runs."""

    tasks = tuple()
    valid_datagroup_keys = ("paths", "labels", "checksums")

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("cache_dir",
            type=str,
            action=ExpandAbspath,
            help="Speechbox cache for storing intermediate output such as extracted features.")
        parser.add_argument("experiment_config",
            type=str,
            action=ExpandAbspath,
            help="Path to a yaml-file containing the experiment configuration, e.g. hyperparameters, feature extractors etc.")
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
                "Error: self.state does not have a 'data' key containing filepaths and labels."
                " Either load an existing dataset definition from the cache with '--load-state' or load a dataset from disk by using a dataset walker."
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

    def exit(self):
        if self.args.save_state:
            self.save_state()
