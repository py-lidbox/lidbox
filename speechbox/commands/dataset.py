import collections
import itertools
import os
import pprint
import sys

from speechbox.commands import ExpandAbspath
from speechbox.commands.base import Command
import speechbox.dataset as dataset
import speechbox.preprocess.transformations as transformations
import speechbox.system as system


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
        "swap_paths_prefix",
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
        parser.add_argument("--swap-paths-prefix",
            type=str,
            action=ExpandAbspath,
            help="Replace the source directory prefix of every path in every datagroups with the given prefix.")
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
        if "sample_frequency" in self.experiment_config:
            walker_config["sample_frequency"] = self.experiment_config["sample_frequency"]
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
        }
        if "sample_frequency" in self.experiment_config:
            parser_config["sample_frequency"] = self.experiment_config["sample_frequency"]
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
        elif args.split == "parse-pre-defined":
            splitter = transformations.dataset_split_parse_predefined
        else:
            splitter = transformations.dataset_split_samples
        split = splitter(dataset_walker, verbosity=args.verbosity)
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
            print("Checking number of files by datagroup.")
        for datagroup_name, datagroup in self.state["data"].items():
            for key in ("paths", "labels", "checksums"):
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
        state_data = self.state["data"]
        if args.verbosity and any("features" in datagroup for datagroup in state_data.values()):
            print("Warning: some datagroups seem to have paths to extracted features. You should re-extract all features after the augmentation is complete.")
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
            flattened_kwargs = [
                [(aug_type, v) for v in aug_values]
                for aug_type, aug_values in all_kwargs
            ]
            # Set augment_config to be all possible combinations of given values
            augment_config = [dict(kwargs) for kwargs in itertools.product(*flattened_kwargs)]
        if args.verbosity > 1:
            print("Full config for augmentation:")
            pprint.pprint(augment_config)
            print()
        print_progress = self.experiment_config.get("print_progress", 0)
        # Collect paths of augmented files by datagroup, each set of augmented paths grouped by the source path it was augmented from
        dst_paths_by_datagroup = {datagroup_key: collections.defaultdict(list) for datagroup_key in state_data}
        num_augmented = 0
        for aug_kwargs in augment_config:
            if args.verbosity:
                print("Augmenting by:")
                for aug_type, aug_value in aug_kwargs.items():
                    print(aug_type, aug_value)
                print()
            for datagroup_key, datagroup in state_data.items():
                src_paths = datagroup["paths"]
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
                        dst_paths_by_datagroup[datagroup_key][src_path].append(dst_path)
                        num_augmented += 1
                        if args.verbosity > 3:
                            print("augmented {} to {}".format(src_path, dst_path))
                    if print_progress > 0 and num_augmented % print_progress == 0:
                        print(num_augmented, "files augmented")
        if print_progress > 0:
            print(num_augmented, "files augmented")
        if args.verbosity:
            print("Adding paths of all augmented files into data state")
        # All augmented, now expand the paths in the state dict
        for datagroup_key, datagroup in state_data.items():
            all_augmented_paths = dst_paths_by_datagroup[datagroup_key]
            addition = {"checksums": [], "labels": [], "paths": []}
            for label, src_path in zip(datagroup["labels"], datagroup["paths"]):
                dst_paths = all_augmented_paths[src_path]
                if not dst_paths:
                    if args.verbosity:
                        print("Warning: path not augmented: '{}'".format(src_path))
                    continue
                addition["checksums"].extend(system.md5sum(path) for path in dst_paths)
                addition["labels"].extend(label for _ in dst_paths)
                addition["paths"].extend(dst_paths)
            for key, vals in addition.items():
                datagroup[key].extend(vals)


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


    def run(self):
        super().run()
        return self.run_tasks()
