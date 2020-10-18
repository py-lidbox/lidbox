"""
lidbox command line interface.
"""
import argparse
import itertools
import logging
import os
import sys

import lidbox
import lidbox.schemas


VERBOSITY_TO_LOGLEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

def create_argparser():
    """Root argparser for all lidbox command line interface commands"""
    description = lidbox.__doc__ + "For help on other commands, try e.g. lidbox e2e -h."
    root_parser = argparse.ArgumentParser(
        prog=lidbox.__name__,
        description=description,
    )
    subparsers = root_parser.add_subparsers(title="subcommands")
    # Create command line options for all valid commands
    for command_group in EXPORT_COMMANDS:
        # Add subparser for this command group
        group_argparser = command_group.create_argparser(subparsers)
        group_argparser.set_defaults(cmd_class=command_group)
    return root_parser


class ExpandAbspath(argparse.Action):
    """Simple argparse action to expand path arguments to full paths using os.path.abspath."""
    def __call__(self, parser, namespace, path, *args, **kwargs):
        setattr(namespace, self.dest, os.path.abspath(path))


class Command:
    """
    Base command class for options shared by all lidbox commands.
    """
    command_name = None
    # All commands have tasks that can be executed, depending on given arguments
    tasks = ()

    @classmethod
    def create_argparser(cls, subparsers):
        parser = subparsers.add_parser(
            cls.command_name or cls.__name__.lower(),
            description=cls.__doc__,
            add_help=False
        )
        required = parser.add_argument_group("required", description="Required arguments")
        required.add_argument("lidbox_config_yaml_path",
            type=str,
            action=ExpandAbspath,
            help="Path to yaml-file, containing the lidbox configuration.")
        optional = parser.add_argument_group(
                "global options",
                description="Optional, global arguments for all subcommands and tasks.")
        optional.add_argument("--run-cProfile",
            action="store_true",
            default=False,
            help="Do profiling on all Python function calls and write results into a file in the working directory.")
        optional.add_argument("--run-tf-profiler",
            action="store_true",
            default=False,
            help="Run the TensorFlow profiler and create a trace event file.")
        optional.add_argument("--help", "-h",
            action="help",
            help="Show this message and exit.")
        optional.add_argument("--verbosity", "-v",
            action="count",
            default=0,
            help="Increases output verbosity. " + ", ".join(
                ["{:d} = {:s}".format(i, logging.getLevelName(l)) for i, l in enumerate(VERBOSITY_TO_LOGLEVEL)] +
                ["3 = tensorflow debug"]))
        return parser

    def __init__(self, args):
        self.args = args

    def run_tasks(self):
        given_tasks = [
            getattr(self, task_name)
            for task_name in self.__class__.tasks
            if getattr(self.args, task_name) and hasattr(self, task_name)
        ]
        if not given_tasks:
            print("Error: No tasks given, doing nothing", file=sys.stderr)
            return 2
        for task_fn in given_tasks:
            # Users might pipe output of some command to POSIX commands e.g. head, tail, cat etc.
            # These might emit SIGPIPE signals back to tell us there is no need to print more data.
            # In Python this is handled by throwing a BrokenPipeError exception.
            # In that case, let's stop without errors and exit with code 0.
            try:
                ret = task_fn()
            except BrokenPipeError:
                return 0
            if ret:
                return ret

    def run(self):
        args = self.args
        max_loglevel = len(VERBOSITY_TO_LOGLEVEL) - 1
        if lidbox.DEBUG:
            print("lidbox.DEBUG is True, overriding given --verbosity setting {} with maximum log level {}".format(args.verbosity, max_loglevel))
            args.verbosity = max_loglevel
        loglevel = min(max_loglevel, max(0, args.verbosity))
        lidbox.reset_global_loglevel(VERBOSITY_TO_LOGLEVEL[loglevel])
        if args.verbosity > 1:
            print("Running lidbox command '{}' with arguments:".format(self.__class__.__name__.lower()))
            lidbox.yaml_pprint(vars(args))
            print()


class E2E(Command):
    """
    Run everything end-to-end as specified in the config file and user script.
    """
    def run(self):
        super().run()
        # Input dynamically to avoid importing TensorFlow when the user e.g. requests just the help string from the command
        import lidbox.api
        args = self.args
        if args.verbosity:
            print("Running end-to-end with config file '{}'".format(args.lidbox_config_yaml_path))
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        split2ds = lidbox.api.create_datasets(split2meta, labels, config)
        assert not any(v is None for v in split2ds.values()), "Empty split, cannot continue"
        history = lidbox.api.run_training(split2ds, config)
        test_conf = config["experiment"]["data"]["test"]
        utt2prediction, utt2target = lidbox.api.predict_with_keras_model(
                split2ds, split2meta, labels, config, test_conf)
        metrics = lidbox.api.evaluate_test_set(
                utt2prediction, utt2target, labels, config, test_conf)
        lidbox.api.write_metrics(metrics, config, test_conf["split"])


class Evaluate(Command):
    """
    Evaluate the test set with a trained model.
    """
    def run(self):
        super().run()
        import lidbox.api
        args = self.args
        if args.verbosity:
            print("Running evaluation with config file '{}'".format(args.lidbox_config_yaml_path))
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        test_split_key = config["experiment"]["data"]["test"]["split"]
        # Run the pipeline only for the test split
        split2meta = {split: meta for split, meta in split2meta.items() if split == test_split_key}
        split2ds = lidbox.api.create_datasets(split2meta, labels, config)
        test_conf = config["experiment"]["data"]["test"]
        utt2prediction, utt2target = lidbox.api.predict_with_keras_model(
                split2ds, split2meta, labels, config, test_conf)
        metrics = lidbox.api.evaluate_test_set(
                utt2prediction, utt2target, labels, config, test_conf)
        lidbox.api.write_metrics(metrics, config, test_conf["split"])


class Prepare(Command):
    """
    Run dataset iterator preparation without training.
    """
    def run(self):
        super().run()
        import lidbox.api
        args = self.args
        if args.verbosity:
            print("Running dataset preparation with config file '{}'".format(args.lidbox_config_yaml_path))
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        split2ds = lidbox.api.create_datasets(split2meta, labels, config)
        print("Prepared iterators:")
        for split, ds in split2ds.items():
            print("   ", split, ds)


class BackendE2E(Command):
    command_name = "backend-e2e"

    def run(self):
        super().run()
        import lidbox.api
        args = self.args
        if args.verbosity:
            print("Extracting embeddings with pre-trained Keras model(s) and training backend classifier(s) on embeddings, using config '{}'".format(args.lidbox_config_yaml_path))
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        split2ds = lidbox.api.create_datasets(split2meta, labels, config)
        lidbox.api.fit_embedding_classifier(
                split2ds, split2meta, labels, config)
        test_conf = config["sklearn_experiment"]["data"]["test"]
        utt2prediction, utt2target = lidbox.api.predict_with_embedding_classifier(
                split2ds, split2meta, labels, config, test_conf)
        metrics = lidbox.api.evaluate_test_set(
                utt2prediction, utt2target, labels, config, test_conf)
        lidbox.api.write_metrics(metrics, config, test_conf["split"])


class BackendPredict(Command):
    command_name = "backend-predict"

    def run(self):
        super().run()
        import lidbox.api
        args = self.args
        if args.verbosity:
            print("Extracting embeddings with pre-trained Keras model(s) and predicting log probabilities with trained backend classifier, using config '{}'".format(args.lidbox_config_yaml_path))
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        data_conf = config["sklearn_experiment"]["data"]["predict"]
        # Drop all other splits, we only want to predict from the data with key 'predict'
        split2meta = {data_conf["split"]: split2meta[data_conf["split"]]}
        split2ds = lidbox.api.create_datasets(split2meta, labels, config)
        utt2prediction, _ = lidbox.api.predict_with_embedding_classifier(
                split2ds, split2meta, labels, config, data_conf)
        lidbox.api.write_predictions(utt2prediction, labels, config, data_conf["split"])


class BackendEvaluate(Command):
    command_name = "backend-evaluate"

    def run(self):
        super().run()
        import lidbox.api
        args = self.args
        if args.verbosity:
            print("Extracting embeddings with pre-trained Keras model(s), predicting log probabilities with trained backend classifier, and evaluating metrics on the test set , using config '{}'".format(args.lidbox_config_yaml_path))
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        test_conf = config["sklearn_experiment"]["data"]["test"]
        split2meta = {test_conf["split"]: split2meta[test_conf["split"]]}
        split2ds = lidbox.api.create_datasets(split2meta, labels, config)
        utt2prediction, utt2target = lidbox.api.predict_with_embedding_classifier(
                split2ds, split2meta, labels, config, test_conf)
        metrics = lidbox.api.evaluate_test_set(
                utt2prediction, utt2target, labels, config, test_conf)
        lidbox.api.write_metrics(metrics, config, test_conf["split"])


class Utils(Command):
    """
    Simple utilities.
    """
    tasks = (
        "load_and_show_meta",
        "run_script",
        "validate_config_file",
    )

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        optional = parser.add_argument_group("utils options")
        optional.add_argument("--load-and-show-meta",
            action="store_true",
            help="Prepare config file for data pipeline evaluation and show the metadata that would be used.")
        optional.add_argument("--validate-config-file",
            action="store_true",
            help="Use a JSON schema to check the given config file is valid.")
        optional.add_argument("--run-script",
            type=str,
            action=ExpandAbspath,
            help="Create a dataset iterator by running the given user script.")
        optional.add_argument("--split",
            type=str,
            help="Use a single, specific split, e.g. 'train', 'test'.")
        return parser

    def _filter_by_splitarg(self, split2meta):
        if self.args.split:
            split2meta = {s: m for s, m in split2meta.items() if s == self.args.split}
        return split2meta

    def load_and_show_meta(self):
        import lidbox.api
        args = self.args
        split2meta, labels, _ = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        split2meta = self._filter_by_splitarg(split2meta)
        for split, meta in split2meta.items():
            print(split)
            for k, v in meta.items():
                print("  ", k, ": ", len(v), " items", sep='')
        print("Total number of labels:", len(labels))

    def validate_config_file(self):
        args = self.args
        errors = lidbox.schemas.validate_config_file_and_get_error_string(args.lidbox_config_yaml_path, args.verbosity)
        if errors:
            print(errors, file=sys.stderr)
            return 1
        elif args.verbosity:
            print("File '{}' ok".format(args.lidbox_config_yaml_path))

    def run_script(self):
        import lidbox.api
        args = self.args
        split2meta, labels, config = lidbox.api.load_splits_from_config_file(args.lidbox_config_yaml_path)
        split2meta = self._filter_by_splitarg(split2meta)
        config["user_script"] = args.run_script
        _ = lidbox.api.create_datasets(split2meta, labels, config)

    def run(self):
        super().run()
        return self.run_tasks()


class Kaldi(Command):
    """
    TODO
    """

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required = parser.add_argument_group("kaldi arguments")
        required.add_argument("wavscp", type=str, action=ExpandAbspath)
        required.add_argument("durations", type=str, action=ExpandAbspath)
        required.add_argument("utt2lang", type=str, action=ExpandAbspath)
        required.add_argument("utt2seg", type=str, action=ExpandAbspath)
        required.add_argument("task", choices=("uniform-segment",))
        optional = parser.add_argument_group("kaldi options")
        optional.add_argument("--offset", type=float)
        optional.add_argument("--window-len", type=float)
        return parser

    def uniform_segment(self):
        def parse_kaldifile(path):
            with open(path) as f:
                for line in f:
                    yield line.strip().split()
        args = self.args
        if os.path.exists(args.utt2seg):
            print("error: segments file already exists, not overwriting: {}".format(args.utt2seg))
            return 1
        file2lang = dict(parse_kaldifile(args.utt2lang))
        utt2seg = {}
        utt2lang = {}
        for fileid, dur in parse_kaldifile(args.durations):
            dur = float(dur)
            num_frames = int(1 + max(0, (dur - args.window_len + args.offset)/args.offset))
            for utt_num in range(num_frames):
                start = utt_num * args.offset
                end = start + args.window_len
                uttid = "{}_{}".format(fileid, utt_num)
                utt2seg[uttid] = (fileid, start, end if end <= dur else -1)
                utt2lang[uttid] = file2lang[fileid]
        # assume mono left
        channel = 0
        with open(args.utt2seg, "w") as f:
            for uttid, (fileid, start, end) in utt2seg.items():
                print(
                    uttid,
                    fileid,
                    format(float(start), ".2f"),
                    format(float(end), ".2f") if end >= 0 else '-1',
                    channel,
                    file=f
                )
        shutil.copyfile(args.utt2lang, args.utt2lang + ".old")
        with open(args.utt2lang, "w") as f:
            for uttid, lang in utt2lang.items():
                print(uttid, lang, file=f)

    def run(self):
        super().run()
        args = self.args
        if args.task == "uniform-segment":
            if args.verbosity:
                print("Creating a segmentation file for all utterances in wav.scp with fixed length segments")
            assert args.offset
            assert args.window_len
            assert args.offset <= args.window_len
            return self.uniform_segment()
        return 1


EXPORT_COMMANDS = (
    E2E,
    Prepare,
    Evaluate,
    BackendE2E,
    BackendPredict,
    BackendEvaluate,
    Kaldi,
    Utils,
)
