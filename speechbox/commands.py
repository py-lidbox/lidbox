"""
Command definitions for all tools.
"""
import argparse
import collections
import itertools
import os
import pprint
import sys

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
            help="Increases verbosity of output for each -v supplied (up to 3).")
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
        if not self.state and self.args.verbosity:
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
                print("Error: key '{}' has an invalid split, some information was probably lost during the split.")
                print("Values in the original data but not in the split:")
                for o in val_original - val_splitted:
                    print(o)
                print("Values in the split but not in the original data:")
                for s in val_splitted - val_original:
                    print(s)
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
        state_path = os.path.join(args.cache_dir, "state.json.gz")
        if args.verbosity:
            print("Loading state from '{}'".format(state_path))
        self.state = system.load_gzip_json(state_path)

    def save_state(self):
        args = self.args
        self.make_named_dir(args.cache_dir, "cache")
        state_path = os.path.join(args.cache_dir, "state.json.gz")
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

    tasks = ("walk", "parse", "split", "check_split", "check_integrity", "to_kaldi", "compare_state", "augment")

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
            choices=dataset.all_split_types,
            help="Create a random training-validation-test split for a dataset and store the splits as datagroups.")
        parser.add_argument("--check-split",
            action="store_true",
            help="Check that all datagroups are disjoint.")
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
            print("Checking that all datagroups are disjoint")
        if not self.state_data_ok():
            return 1
        datagroups = self.state["data"]
        if len(datagroups) == 1 and args.verbosity:
            print("Only one datagroup found: '{}', doing nothing".format(list(datagroups.keys())[0]))
            return 1
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
                print("error: Datagroups are not disjoint, following files have equal checksums:")
                for paths in duplicates:
                    for path in paths:
                        print(path)
            else:
                print("ok")

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
        if "all" in self.state.get("data", {}):
            if args.verbosity:
                print("Using existing dataset paths from current state, possibly loaded from the cache")
                if args.src:
                    print("Ignoring dataset source directory '{}'".format(args.src))
            walker_config = self.merge_datagroups_from_state()
        else:
            if args.verbosity:
                print("Dataset paths not found in self.state['data'], walking over whole dataset to gather paths")
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
                if args.verbosity:
                    print("Error: Dataset walker '{}' does not support parsing speaker ids from audio file paths, cannot create 'utt2spk'. Define a parse_speaker_id for the walker class.".format(str(dataset_walker)))
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

    #TODO this is more like 'transform dataset from --src to --dst' since it does not add the output paths into self.state
    def augment(self):
        if not self.args_dst_ok() or not self.has_state():
            return 1
        if "source_directory" not in self.state:
            print("Error: no source directory in cache, run --walk and --save-state on some dataset in directory specified by --src to gather all paths into the cache.", file=sys.stderr)
            return 1
        args = self.args
        prefix = self.state["source_directory"]
        if args.verbosity:
            print("Augmenting dataset from '{}' into '{}' with parameters:".format(prefix, args.dst))
        sox_config = self.experiment_config["sox_transform"]
        if args.verbosity:
            pprint.pprint(sox_config)
        src_paths = list(itertools.chain(*(d["paths"] for d in self.state["data"].values())))
        # Retain directory structure after prefix of original paths when writing to the target directory
        dst_paths = [p.replace(prefix, args.dst) for p in src_paths]
        for path in dst_paths:
            self.make_named_dir(os.path.dirname(path))
        augmented_paths = []
        print_progress = self.experiment_config.get("print_progress", 0)
        if args.verbosity:
            print("Starting augmentation")
        for i, path in enumerate(system.apply_sox_transformer(src_paths, dst_paths, sox_config), start=1):
            augmented_paths.append(path)
            if print_progress and i % print_progress == 0:
                print(i, "files done")
        if args.verbosity:
            if len(augmented_paths) != len(src_paths):
                print("Error: failed to apply transformation.")
            print("Out of {} input paths, {} were augmented.".format(len(src_paths), len(augmented_paths)))

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


class Train(Command):
    """Model training."""

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--load-model",
            action="store_true",
            help="Load pre-trained model from cache directory.")
        parser.add_argument("--save-model",
            action="store_true",
            help="Save model to the cache directory, overwriting any existing models with the same name.")
        parser.add_argument("--model-id",
            type=str,
            help="Use this value as the model name instead of the one in the experiment yaml-file.")
        return parser

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
        training_set, training_set_meta = system.load_features_as_dataset(
            # List of all .tfrecord files containing all training set samples
            [data["training"]["features"]],
            model_config
        )
        # Same for the validation set
        validation_set, _ = system.load_features_as_dataset(
            [data["validation"]["features"]],
            model_config
        )
        model.prepare(training_set_meta, model_config)
        if args.verbosity:
            print("\nStarting training\n")
        model.fit(training_set, validation_set, model_config)
        if args.verbosity:
            print("\nTraining finished\n")

    def run(self):
        super().run()
        args = self.args
        self.model_id = args.model_id if args.model_id else self.experiment_config["model"]["name"]
        if args.load_model:
            if args.verbosity:
                print("Loading model '{}' from the cache directory".format(self.model_id))
            self.state["model"] = models.KerasWrapper.from_disk(args.cache_dir, self.model_id)
        else:
            if args.verbosity:
                print("Creating new model '{}'".format(self.model_id))
            self.state["model"] = models.KerasWrapper(self.model_id)
        return self.train()

    def exit(self):
        args = self.args
        if args.save_model:
            if "model" not in self.state:
                print("Error: no model to save")
                return 1
            saved_path = self.state["model"].to_disk(args.cache_dir)
            if args.verbosity:
                print("Wrote model as '{}'".format(saved_path))
        super().exit()


class Evaluate(Command):
    """Prediction and evaluation using trained models."""

    tasks = ("evaluate_test_set",)

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("--model-id",
            type=str,
            help="Use this value as the model name instead of the one in the experiment yaml-file.")
        parser.add_argument("--evaluate-test-set",
            action="store_true",
            help="Evaluate model on test set")
        return parser

    def evaluate_test_set(self):
        args = self.args
        if args.verbosity:
            print("Preparing model for evaluation")
        if not self.has_state() or "test" not in self.state["data"]:
            if args.verbosity:
                print("Error: test set paths not found")
            return 1
        self.model_id = args.model_id if args.model_id else self.experiment_config["model"]["name"]
        if args.verbosity:
            print("Loading model '{}' from the cache directory".format(self.model_id))
        model = models.KerasWrapper.from_disk(args.cache_dir, self.model_id)
        model_config = self.experiment_config["model"]
        test_set, _ = system.load_features_as_dataset(
            [self.state["data"]["test"]["features"]],
            model_config
        )
        model.evaluate(test_set, model_config)

    def run(self):
        super().run()
        return self.run_tasks()


all_commands = (
    Dataset,
    Preprocess,
    Train,
    Evaluate,
)
