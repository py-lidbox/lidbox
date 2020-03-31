import itertools
import os
import pprint
import shutil
import sys
import time

import jsonschema
import numpy as np

from lidbox.commands.base import Command, BaseCommand, ExpandAbspath
import lidbox
import lidbox.system as system
import lidbox.visualization as visualization


class Util(BaseCommand):
    """CLI for IO functions in lidbox.system."""
    tasks = (
        "yaml_get",
        "get_unique_duration",
        "plot_melspectrogram",
        "watch",
        "assert_disjoint",
        "validate_config_files",
    )

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required = parser.add_argument_group("util arguments")
        required.add_argument("infile",
            type=str,
            nargs="*",
            help="Paths to one or more input files to apply the command for")
        optional = parser.add_argument_group("util options")
        optional.add_argument("--yaml-get",
            type=str,
            metavar="KEY",
            help="Parse yaml files and dump contents of given yaml key to stdout.")
        optional.add_argument("--get-unique-duration",
            action="store_true",
            help="Group all files by MD5 sums and compute total duration for all unique, valid audio files. Files that cannot be opened as audio files are ignored. If there are duplicate files by MD5 sum, the first file is chosen in the same order as it was given as argument to this command.")
        optional.add_argument("--plot-melspectrogram",
            action="store_true",
            help="Plot mel spectrograms for all given files. Blocks control until all figures are manually closed.")
        optional.add_argument("--validate-config-files",
            action="store_true",
            help="Validate one or more config files against the config file JSON schema.")
        optional.add_argument("--watch",
            action="store_true")
        optional.add_argument("--transform",
            type=str,
            action=ExpandAbspath,
            help="Apply transformations from this experiment config file  as they would be performed before feature extraction. Then play audio and visualize spectrograms.")
        optional.add_argument("--assert-disjoint",
            action="store_true",
            help="Read list of paths from each infile, reduce paths to basenames, and assert that each list of basenames is disjoint with the basename lists from other files")
        return parser

    @staticmethod
    def get_yaml_key(path, keys):
        value = system.load_yaml(path)
        for key in keys.split('.'):
            value = value[key]
        return value

    def yaml_get(self):
        args = self.args
        for infile in args.infile:
            value = self.get_yaml_key(infile, args.yaml_get)
            if type(value) is list:
                value_str = '\n'.join(value)
            elif type(value) is dict:
                value_str = pprint.pformat(value, indent=1)
            else:
                value_str = str(value)
            print(value_str)

    def get_unique_duration(self):
        args = self.args
        if args.verbosity:
            print("Calculating total duration for {} files".format(len(args.infile)))
        seen_files = set()
        valid_files = []
        for infile in args.infile:
            md5sum = system.md5sum(infile)
            if md5sum in seen_files:
                continue
            seen_files.add(md5sum)
            if system.read_wavfile(infile)[0] is not None:
                valid_files.append(infile)
        del seen_files
        if args.verbosity:
            print("Found {} invalid or duplicate files, they will be ignored".format(len(args.infile) - len(valid_files)))
            print("Calculating total duration for {} valid files".format(len(valid_files)))
        print(system.format_duration(system.get_total_duration(valid_files)))

    def plot_melspectrogram(self):
        args = self.args
        if args.verbosity:
            print("Plotting mel spectrograms for {} files".format(len(args.infile)))
        for filenum, infile in enumerate(args.infile, start=1):
            wav = system.read_wavfile(infile)
            if wav[0] is None:
                if args.verbosity:
                    print("Invalid file, cannot open as wav: {}".format(infile))
                continue
            visualization.plot_melspectrogram(wav, filenum, block=False)
        if args.verbosity:
            print("All plotted, waiting for user to close all figures.")
        visualization.show()

    def watch(self):
        args = self.args
        if args.verbosity:
            print("Watching {} files".format(len(args.infile)))
            if args.watch:
                print("Waiting for {} seconds in between files".format(args.watch))
            else:
                print("Press r for repeat, any key for next file")
        for filenum, infile in enumerate(args.infile, start=1):
            wav = system.read_wavfile(infile)
            if wav[0] is None:
                if args.verbosity:
                    print("Invalid file, cannot open as wav: {}".format(infile))
                continue
            visualization.plot_overview(wav, "/tmp/plot.svg")
            print(infile)
            repeat = True
            while repeat:
                err = next(system.run_for_files("play --no-show-progress", [infile]))
                if args.watch:
                    time.sleep(args.watch)
                    repeat = False
                elif input().strip().lower() != 'r':
                    repeat = False
                if err and args.verbosity:
                    print(err, file=sys.stderr)

    def assert_disjoint(self):
        args = self.args
        if args.verbosity:
            print("Checking that filelists contain disjoint sets of files")
        path_lists = []
        for infile in args.infile:
            with open(infile) as f:
                path_lists.append((infile, [line.strip().split()[0] for line in f]))
        for (f1, l1), (f2, l2) in itertools.combinations(path_lists, r=2):
            print(f1, "vs", f2, " ... ", end='')
            common = set(os.path.basename(p) for p in l1) & set(os.path.basename(p) for p in l2)
            if common:
                print("fail, {} basenames in common".format(len(common)))
                if args.verbosity > 2:
                    for f in common:
                        print(f)
            else:
                print("ok")

    def validate_config_files(self):
        args = self.args
        if args.verbosity:
            print("Validating configuration files against JSON schema from '{}'.".format(lidbox.CONFIG_FILE_SCHEMA_PATH))
        schema = system.load_yaml(lidbox.CONFIG_FILE_SCHEMA_PATH)
        for infile in args.infile:
            config = system.load_yaml(infile)
            try:
                jsonschema.validate(instance=config, schema=schema)
                print("File '{}' ok".format(infile))
            except jsonschema.ValidationError as error:
                print("File '{}' validation failed, error is:\n  {}".format(infile, error.message))
                if error.context:
                    print("context:")
                    for context in error.context:
                        print(context)
                if error.cause:
                    print("cause:\n", error.cause)
                if args.verbosity > 1:
                    print("Instance was:")
                    lidbox.yaml_pprint(error.instance, left_pad=2)

    def run(self):
        return self.run_tasks()


class Kaldi(BaseCommand):

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


command_tree = [
    (Util, []),
    (Kaldi, []),
]
