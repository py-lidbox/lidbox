import collections
import itertools
import os
import pprint
import sys

import kaldiio
import numpy as np

from speechbox.commands.base import State, Command, StatefulCommand, ExpandAbspath
import speechbox.dataset as dataset
import speechbox.preprocess.transformations as transformations
import speechbox.system as system


def parse_lines(path):
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                yield l.split()


class Dataset(Command):
    """
    Speech dataset and corpus parser for loading absolute paths of samples from disk.
    """
    tasks = tuple()
    # (
    #     "to_kaldi",
    #     "swap_paths_prefix",
    # )
    valid_datagroup_keys = ("paths", "labels")

    def merge_datagroups_from_state(self):
        data = self.state["data"]
        return {
            key: list(itertools.chain(*(datagroup[key] for datagroup in data.values())))
            for key in self.valid_datagroup_keys
        }

    def to_kaldi(self):
        args = self.args
        kaldi_dir = args.to_kaldi
        self.make_named_dir(kaldi_dir, "kaldi output")
        for datagroup_name, datagroup in self.state["data"].items():
            paths, labels = datagroup["paths"], datagroup["labels"]
            if args.verbosity:
                print("'{}', containing {} paths".format(datagroup_name, len(paths)))
            dataset_walker = dataset.get_dataset_walker(self.dataset_id, {"paths": paths, "labels": labels})
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

    def swap_paths_prefix(self):
        args = self.args
        if not self.state_data_ok():
            return 1
        prefix = self.state["source_directory"]
        new_prefix = args.swap_paths_prefix
        if args.verbosity:
            print("Swapping prefix '{}' for '{}' in all paths in state".format(prefix, new_prefix))
        for datagroup in self.state["data"].values():
            datagroup["paths"] = [p.replace(prefix, new_prefix) for p in datagroup["paths"]]
        self.state["source_directory"] = new_prefix


class Gather(StatefulCommand):
    """Gather all filepaths from a dataset into experiment state"""
    tasks = (
        "walk_dir",
        "load_from_list",
        "load_kaldi_files",
    )
    requires_state = State.none

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        tasks = parser.add_argument_group("gather tasks", description="Different ways to gather filepaths from a dataset")
        tasks.add_argument("--walk-dir",
            type=str,
            action=ExpandAbspath,
            metavar="DATASET_DIR",
            help="Walk over a dataset starting at this directory, gathering audio file paths and labels. All paths and labels are written into experiment state under key 'all'.")
        tasks.add_argument("--load-from-list",
            type=str,
            action=ExpandAbspath,
            metavar="PATH_LIST",
            help="Load path list from this file. Every line must contain atleast two columns, separated by space or tab, such that the first column is an absolute path and the second column is the label.")
        tasks.add_argument("--load-kaldi-files",
            action="store_true",
            help="Load all paths, labels and pre-extracted features from kaldi files specified in the config file.")
        optional = parser.add_argument_group("gather options")
        optional.add_argument("--check",
            action="store_true",
            help="Check that every file is unique and contains audio. Might take a long time since every file will be opened briefly with sox and/or librosa.")
        # optional.add_argument("--compute-checksums",
            # action="store_true",
            # help="Compute MD5 checksums of each file during gather.")
        optional.add_argument("--datagroup",
            type=str,
            help="Which key to use for datagroup when loading paths with --load-from-list.")
        return parser

    def kaldi_files_ok(self):
        ok = True
        config = self.experiment_config["features"].get("kaldi")
        if config is None:
            print("Error: kaldi files need to be defined in the config file in a list at key features.kaldi")
            ok = False
        else:
            for kaldi in config:
                for f in ("datagroup", "feats-ark", "wav-scp", "utt2label"):
                    if f not in kaldi:
                        print("Error: '{}' not defined in config file".format(f))
                        ok = False
                    elif f != "datagroup" and not os.path.exists(kaldi[f]):
                        print("Error: '{}' does not exist: '{}'".format(f, kaldi[f]))
                        ok = False
        return ok

    def walk_dir(self):
        args = self.args
        if args.verbosity:
            print("Walking over dataset '{}'".format(self.dataset_id))
        if self.state:
            if args.verbosity:
                print("Using paths from state, gathered during some previous walk")
            walker_config = self.merge_datagroups_from_state()
        else:
            src = args.walk_dir
            walker_config = {
                "dataset_root": src
            }
            if args.verbosity:
                print("Gathering paths from directory '{}'.".format(src))
        walker_config["enabled_labels"] = self.experiment_config["dataset"]["labels"]
        sample_frequency = self.experiment_config["dataset"].get("sample_frequency")
        if sample_frequency:
            walker_config["sample_frequency"] = sample_frequency
        dataset_walker = dataset.get_dataset_walker(self.dataset_id, walker_config)
        if args.check:
            if args.verbosity:
                print("'--check' given, invalid files will be ignored during the walk")
            dataset_iter = dataset_walker.walk(check_read=True, check_duplicates=True, verbosity=args.verbosity)
        else:
            if args.verbosity:
                print("'--check' not given, walk will be much faster but invalid files will not be ignored")
            dataset_iter = dataset_walker.walk(verbosity=args.verbosity)
        data = self.state.get("data", {})
        if args.verbosity and "all" in data:
            print("Warning: overwriting existing paths in self.state for dataset '{}'".format(self.dataset_id))
        # Gather all
        labels, paths = [], []
        for label, path in dataset_iter:
            labels.append(label)
            paths.append(path)
        walk_data = {
            "label_to_index": dataset_walker.make_label_to_index_dict(),
            "labels": labels,
            "paths": paths,
        }
        # Merge to state under key 'all'
        self.state["data"] = dict(data, all=walk_data)
        self.state["source_directory"] = self.experiment_config["dataset"]["src"]
        self.state["state"] = State.has_paths

    def load_from_list(self):
        args = self.args
        path_list = args.load_from_list
        if not args.datagroup:
            print("Error: --datagroup required with --load-from-list")
            return 1
        if args.verbosity:
            print("Gathering paths from list in file '{}' into datagroup '{}'.".format(path_list, args.datagroup))
        if "label_to_index" not in self.state:
            if args.verbosity:
                print("label_to_index not found, using order of the label list from the config file as a label_to_index mapping")
            self.state["label_to_index"] = {label: i for i, label in enumerate(self.experiment_config["dataset"]["labels"])}
        paths, labels = system.parse_path_list(path_list)
        if "data" not in self.state:
            self.state["data"] = {}
        elif args.datagroup in self.state["data"]:
            if args.verbosity:
                print("Warning: Overwriting existing datagroup '{}' in state".format(args.datagroup))
        self.state["data"][args.datagroup] = {
            "paths": paths,
            "labels": labels,
        }
        self.state["state"] = State.has_paths
        # if args.compute_checksums:
        #     self.state["data"][args.datagroup]["checksums"] = system.all_md5sums(paths)

    def load_kaldi_files(self):
        args = self.args
        if args.verbosity:
            print("Loading pre-extracted Kaldi-feats")
        if not self.kaldi_files_ok():
            return 1
        label_to_index = {label: i for i, label in enumerate(self.experiment_config["dataset"]["labels"])}
        config = self.experiment_config["features"]
        if "data" not in self.state:
            self.state["data"] = {}
        for kaldi_paths in config["kaldi"]:
            wavdata = collections.defaultdict(dict)
            for utt, path in parse_lines(kaldi_paths["wav-scp"]):
                wavdata[utt]["path"] = path
            for utt, label in parse_lines(kaldi_paths["utt2label"]):
                wavdata[utt]["label"] = label
            for utt, features in kaldiio.load_ark(kaldi_paths["feats-ark"]):
                if kaldi_paths["datagroup"] == "training":
                    utt = utt[3:]
                else:
                    utt = utt[4:]
                wavdata[utt]["features"] = features
            ok = True
            for utt, data in wavdata.items():
                missing_keys = []
                for k in ("path", "label", "features"):
                    if k not in data:
                        missing_keys.append(k)
                if missing_keys:
                    ok = False
                    print("Error: utterance '{}' is missing keys: {}".format(utt, ', '.join(missing_keys)))
            if not ok:
                return 1
            features_by_label = collections.defaultdict(list)
            paths = []
            labels = []
            for utt in wavdata.values():
                paths.append(utt["path"])
                label, feats = utt["label"], utt["features"]
                labels.append(label)
                onehot = np.zeros(len(label_to_index), dtype=np.float32)
                onehot[label_to_index[label]] = 1.0
                features_by_label[label].append((features, onehot))
            del wavdata
            datagroup_name = kaldi_paths["datagroup"]
            datagroup = {
                "paths": paths,
                "labels": labels,
                "features": {},
            }
            tfrecords_dir = os.path.join(self.cache_dir, datagroup_name)
            self.make_named_dir(tfrecords_dir, "features")
            sequence_length = config.get("sequence_length", 0)
            for label, features in features_by_label.items():
                features = iter(features)
                if args.verbosity:
                    print("Writing kaldi features as TFRecord files for label '{}'".format(label))
                output_path = os.path.join(tfrecords_dir, label)
                if sequence_length > 0:
                    wrote_path = system.write_sequence_features(features, output_path, sequence_length)
                else:
                    wrote_path = system.write_features(features, output_path)
                datagroup["features"][label] = wrote_path
                if args.verbosity > 1:
                    print("Wrote '{}' features for label '{}' into '{}'".format(datagroup_name, label, wrote_path))
            self.state["data"][datagroup_name] = datagroup
        self.state["label_to_index"] = label_to_index
        self.state["state"] = State.has_features

    def run(self):
        super().run()
        return self.run_tasks()


class Split(StatefulCommand):
    """Partition dataset paths into training-test splits or check validity of existing splits."""
    tasks = (
        "random",
        "check",
    )
    requires_state = State.has_paths

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        required = parser.add_argument_group("split arguments")
        required.add_argument("split_type",
            type=str,
            choices=dataset.all_split_types,
            help="How to split the dataset paths")
        optional = parser.add_argument_group("split options")
        optional.add_argument("--random",
            action="store_true",
            help="Create a random training-validation-test split from all paths in datagroup 'all' and store the split into state as new datagroups. This will delete the datagroup 'all'.")
        optional.add_argument("--check",
            action="store_true",
            help="Check that all datagroups are disjoint by the given split type.")
        optional.add_argument("--ratio",
            type=float,
            default=0.1,
            help="Test and validation set size as a fraction of the whole dataset.")
        return parser

    @staticmethod
    def split_is_a_partition(original, split):
        """
        Check that all values from 'original' exist in 'split' and all values in 'split' are from 'original'.
        """
        ok = True
        train, val, test = split["training"], split["validation"], split["test"]
        for key in Dataset.valid_datagroup_keys:
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

    def random(self):
        args = self.args
        if args.verbosity:
            print("Creating a random training-validation-test split for dataset '{}' using split type '{}'".format(self.dataset_id, args.split_type))
        datagroups = self.state["data"]
        if len(datagroups) > 1:
            error_msg = (
                "Error: The loaded state already has a split into {} different datagroups:\n".format(len(datagroups)) +
                '\n'.join(str(datagroup) for datagroup in datagroups)
            )
            print(error_msg, file=sys.stderr)
            return 1
        if "all" not in datagroups:
            print("Error, no key 'all' in datagroups, cannot split", file=sys.stderr)
            return 1
        dataset_config = self.experiment_config["dataset"]
        # We need a dataset walker for parsing speaker ids from paths
        walker_config = {
            "dataset_root": dataset_config["src"],
            "enabled_labels": dataset_config["labels"],
        }
        dataset_walker = dataset.get_dataset_walker(self.dataset_id, walker_config)
        # The label-to-index mapping is common for all datagroups in the split
        self.state["label_to_index"] = dataset_walker.make_label_to_index_dict()
        data = datagroups["all"]
        samples = list(zip(data["paths"], data["labels"]))
        if args.split_type == "by-speaker":
            split = transformations.dataset_split_samples_by_speaker(
                samples,
                dataset_walker.parse_speaker_id,
                validation_ratio=args.ratio,
                test_ratio=args.ratio,
                verbosity=args.verbosity
            )
        else:
            split = transformations.dataset_split_samples(
                samples,
                validation_ratio=args.ratio,
                test_ratio=args.ratio,
                verbosity=args.verbosity
            )
        split_data = {}
        for key, datagroup in split.items():
            paths, labels = [], []
            for path, label in datagroup:
                paths.append(path)
                labels.append(label)
            split_data[key] = {
                "paths": paths,
                "labels": labels,
            }
        if args.verbosity:
            print("Checking that the split is a partition of the original data from key 'all'")
        if not self.split_is_a_partition(data, split_data):
            return 1
        self.state["data"] = split_data

    def check(self):
        args = self.args
        if args.verbosity:
            print("Checking that all datagroups are disjoint by split type '{}'".format(args.split_type))
        if not self.state_data_ok():
            return 1
        datagroups = self.state["data"]
        if len(datagroups) == 1:
            print("Error: Only one datagroup found: '{}', cannot check that it is disjoint with other datagroups since other datagroups do not exist".format(list(datagroups.keys())[0]))
            return 1
        if args.split_type == "by-file":
            for a, b in itertools.combinations(datagroups.keys(), r=2):
                print("'{}' vs '{}' ... ".format(a, b), flush=True, end='')
                # Group all filepaths by MD5 checksums of file contents
                duplicates = collections.defaultdict(list)
                a_paths = datagroups[a]["paths"]
                a_paths = zip(system.all_md5sums(a_paths), a_paths)
                b_paths = datagroups[b]["paths"]
                b_paths = zip(system.all_md5sums(b_paths), b_paths)
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
        elif args.split_type == "by-speaker":
            dataset_config = self.experiment_config["dataset"]
            walker_config = {
                "dataset_root": dataset_config["src"],
                "enabled_labels": dataset_config["labels"],
            }
            parse_speaker_id = dataset.get_dataset_walker(self.dataset_id, walker_config).parse_speaker_id
            for a, b in itertools.combinations(datagroups.keys(), r=2):
                print("'{}' vs '{}' ... ".format(a, b), flush=True, end='')
                # Group all filepaths by speaker ids parsed from filepaths of the given dataset
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
            print("Error: Unknown split type '{}', cannot check".format(args.split_type), file=sys.stderr)
            return 1

    def check_integrity(self):
        if not self.has_state():
            return 1
        args = self.args
        if args.verbosity > 1:
            print("Number of files by datagroup:")
            for datagroup_name, datagroup in self.state["data"].items():
                for key in self.valid_datagroup_keys:
                    print(datagroup_name, key, len(datagroup.get(key, [])))
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
        if args.verbosity:
            for datagroup_name, mismatch_list in mismatches.items():
                print("Datagroup '{}', {} files:".format(datagroup_name, len(mismatch_list)))
                print("old", "new", "path")
                for old, new, path in mismatch_list:
                    print("{} {} {}".format(old, new, path))

    def run(self):
        super().run()
        if not self.state_ok():
            return 1
        return self.run_tasks()


class Inspect(StatefulCommand):
    """Perform non-modifying actions on dataset paths loaded into state."""
    tasks = (
        "count_files",
        "dump_datagroup",
        "get_audio_durations",
    )
    requires_state = State.has_paths

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        tasks = parser.add_argument_group("inspection tasks", description="Different ways to inspect the dataset")
        tasks.add_argument("--count-files",
            action="store_true",
            help="Count files by label of all datagroups and write results to stdout.")
        tasks.add_argument("--dump-datagroup",
            type=str,
            metavar="DATAGROUP_KEY",
            help="Write all paths and label pairs of given datagroup into stdout.")
        tasks.add_argument("--get-audio-durations",
            action="store_true",
            help="Run SoXi on all paths in all datagroups to compute durations of wav-files by label.")
        return parser

    @staticmethod
    def paths_by_label(d):
        label_key_fn = lambda pair: pair[0]
        paths_sorted_by_label = sorted(zip(d["labels"], d["paths"]), key=label_key_fn)
        return itertools.groupby(paths_sorted_by_label, label_key_fn)

    def count_files(self):
        args = self.args
        if not self.state_data_ok():
            return 1
        if args.verbosity:
            print("Computing amount of files by label")
        for datagroup_key, datagroup in self.state["data"].items():
            print("datagroup '{}'".format(datagroup_key))
            for label, group in self.paths_by_label(datagroup):
                print("  {:s}: {:d}".format(label, len(list(group))))

    def dump_datagroup(self):
        args = self.args
        if not self.state_data_ok():
            return 1
        if args.verbosity:
            print("Writing datagroup '{}' paths and labels into stdout".format(args.dump_datagroup))
        datagroup = self.state["data"][args.dump_datagroup]
        for path, label in zip(datagroup["paths"], datagroup["labels"]):
            print(path, label)

    def get_audio_durations(self):
        args = self.args
        if not self.state_data_ok():
            return 1
        if args.verbosity:
            print("Computing durations of all audio files")
        for datagroup_key, datagroup in self.state["data"].items():
            print("datagroup '{}'".format(datagroup_key))
            for label, group in self.paths_by_label(datagroup):
                paths = [path for label, path in group]
                duration_str = system.format_duration(system.get_total_duration(paths))
                print("  {:s}: {duration_str:s}".format(label, duration_str=duration_str))

    def run(self):
        super().run()
        if not self.state_ok():
            return 1
        return self.run_tasks()


class Augment(StatefulCommand):
    """
    Create new samples for a dataset by applying audio transformations on existing wav-files.
    """
    requires_state = State.has_paths

    def augment(self):
        args = self.args
        if args.verbosity:
            print("Augmenting dataset")
        if not self.state_data_ok():
            return 1
        data = self.state["data"]
        if args.verbosity and any("features" in datagroup for datagroup in data.values()):
            print("Warning: some datagroups seem to have paths to extracted features. You should re-extract all features in case some datagroup gets new files by augmentation.")
        augment_config = self.experiment_config["augmentation"]
        if "output_dir" in augment_config:
            output_dir = os.path.abspath(augment_config["output_dir"])
        else:
            output_dir = os.path.join(self.cache_dir, "augmented-data")
        self.make_named_dir(output_dir, "augmentation output")
        if args.verbosity > 1:
            print("Writing augmentation output into '{}'".format(output_dir))
        augment_params = []
        if "list" in augment_config:
            augment_params = [list(d.items()) for d in augment_config["list"]]
        elif "cartesian_product" in augment_config:
            all_kwargs = augment_config["cartesian_product"].items()
            flattened_kwargs = [
                [(aug_type, v) for v in aug_values]
                for aug_type, aug_values in all_kwargs
            ]
            # Set augment_params to be all possible combinations of given values
            augment_params = list(itertools.product(*flattened_kwargs))
        if args.verbosity > 1:
            print("Full config for augmentation:")
            pprint.pprint(augment_params)
            print()
        print_progress = self.experiment_config.get("print_progress", 0)
        augment_datagroups = augment_config.get("datagroups", list(data.keys()))
        if args.verbosity > 1:
            print("Datagroups that will be augmented:")
            pprint.pprint(augment_datagroups)
            print()
        # Collect paths of augmented files by datagroup,
        # each set of augmented paths grouped by the source path it was augmented from
        dst_paths_by_datagroup = {datagroup_key: collections.defaultdict(list) for datagroup_key in augment_datagroups}
        num_augmented = 0
        for transform_steps in augment_params:
            if args.verbosity > 1:
                print("Augmenting by:")
                for aug_type, aug_value in transform_steps:
                    print(aug_type, aug_value)
                print()
            augdir = '__'.join((str(aug_type) + '_' + str(aug_value)) for aug_type, aug_value in transform_steps)
            for datagroup_key in augment_datagroups:
                datagroup = data[datagroup_key]
                src_paths = datagroup["paths"]
                # Create sox src-dst file pairs for each transformation
                dst_paths = []
                for src_path in src_paths:
                    # Make dirs if they do not exist
                    target_dir = os.path.join(output_dir, augdir)
                    self.make_named_dir(target_dir)
                    dst_paths.append(os.path.join(target_dir, os.path.basename(src_path)))
                for src_path, dst_path in system.apply_sox_transformer(src_paths, dst_paths, transform_steps):
                    if dst_path is None:
                        print("Warning, sox failed to transform '{}' from '{}'".format(dst_path, src_path), file=sys.stderr)
                    else:
                        dst_paths_by_datagroup[datagroup_key][src_path].append(dst_path)
                        num_augmented += 1
                        if args.verbosity > 3:
                            print("augmented {} to {}".format(src_path, dst_path))
                    if args.verbosity and print_progress > 0 and num_augmented % print_progress == 0:
                        print(num_augmented, "files done")
        if args.verbosity:
            if print_progress > 0:
                print(num_augmented, "files done")
            print("Adding paths of all augmented files into data state")
        # All augmented, now expand the paths in the state dict
        for datagroup_key in augment_datagroups:
            datagroup = data[datagroup_key]
            all_augmented_paths = dst_paths_by_datagroup[datagroup_key]
            addition = {"labels": [], "paths": []}
            for label, src_path in zip(datagroup["labels"], datagroup["paths"]):
                dst_paths = all_augmented_paths[src_path]
                if not dst_paths:
                    print("Warning: path not augmented: '{}'".format(src_path), file=sys.stderr)
                    continue
                addition["labels"].extend(len(dst_paths)*[label])
                addition["paths"].extend(dst_paths)
            for key, vals in addition.items():
                datagroup[key].extend(vals)

    def run(self):
        super().run()
        if not self.state_ok():
            return 1
        return self.augment()


class Parse(Command):
    requires_state = State.none

    @classmethod
    def create_argparser(cls, parent_parser):
        parser = super().create_argparser(parent_parser)
        required = parser.add_argument_group("parse arguments")
        required.add_argument("dataset",
            type=str,
            choices=dataset.all_parsers,
            help="ID of dataset to parse.")
        required.add_argument("src",
            type=str,
            action=ExpandAbspath,
            help="Parse files from this directory.")
        required.add_argument("dst",
            type=str,
            action=ExpandAbspath,
            help="Save parsed output into this directory.")
        optional = parser.add_argument_group("parse options")
        optional.add_argument("--resample-to",
            type=int,
            help="Resample all files to this sampling rate.")
        optional.add_argument("--min-duration-ms",
            type=int,
            help="Drop all audio files shorter than the given limit in milliseconds.")
        optional.add_argument("--limit",
            type=int,
            help="Only parse this many audio files, starting from the files with the highest |num_upvotes - num_downvotes| value.")
        optional.add_argument("--duration-limit-sec",
            type=int,
            help="Maximum total duration of all parsed files. I.e. stop parsing when this many seconds of audio have been parsed.")
        optional.add_argument("--normalize-volume",
            type=float,
            help="Use SoX to normalize volume of parsed files to given dBFS, e.g. -3.0.")
        optional.add_argument("--fail-early",
            action="store_true",
            default=False,
            help="Stop parsing when an error occurs instead of skipping over the error file.")
        return parser

    def parse(self):
        args = self.args
        if args.verbosity:
            print("Parsing dataset '{}' from '{}' into '{}'".format(args.dataset, args.src, args.dst))
        self.make_named_dir(args.dst, "parse output")
        parser_config = {
            "dataset_root": args.src,
            "verbosity": args.verbosity,
            "output_dir": args.dst,
            "resampling_freq": args.resample_to,
            "min_duration_ms": args.min_duration_ms,
            "output_count_limit": args.limit,
            "output_duration_limit": args.duration_limit_sec,
            "fail_early": args.fail_early,
        }
        if args.normalize_volume is not None:
            assert args.normalize_volume <= 0, "Expected normalization parameter to be negative, i.e. decibels relative to full scale, but {} was given".format(args.normalize_volume)
            parser_config["normalize_volume"] = args.normalize_volume
        parser = dataset.get_dataset_parser(args.dataset, parser_config)
        if args.verbosity:
            print("Starting parse with parser", repr(parser))
        num_parsed = 0
        if not args.verbosity:
            for path, output in parser.parse():
                if output is not None:
                    num_parsed += 1
        else:
            for path, output in parser.parse():
                if output is None:
                    print("Warning: failed to parse '{}'".format(path), file=sys.stderr)
                    continue
                num_parsed += 1
                if args.verbosity > 1 and any(output):
                    status, out, err = output
                    msg = "Warning:"
                    if status:
                        msg += " exit code: {}".format(status)
                    if out:
                        print(msg, out, file=sys.stderr)
                    if err:
                        print(msg, err, file=sys.stderr)
                if num_parsed % 1000 == 0:
                    print(num_parsed, "files parsed")
            print(num_parsed, "files parsed from '{}' to '{}'".format(args.src, args.dst))

    def run(self):
        super().run()
        return self.parse()


command_tree = [
    (Dataset, [Gather, Split, Inspect, Augment, Parse]),
]
