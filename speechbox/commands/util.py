import itertools
import os
import pprint
import shutil
import sys
import time

import numpy as np

from speechbox.commands.base import State, Command, StatefulCommand, ExpandAbspath
import speechbox.system as system
import speechbox.visualization as visualization


class Util(Command):
    """CLI for IO functions in speechbox.system."""
    tasks = (
        "yaml_get",
        "get_unique_duration",
        "plot_melspectrogram",
        "watch",
        "assert_disjoint",
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


    def run(self):
        return self.run_tasks()


class Plot(StatefulCommand):
    requires_state = State.has_features

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required = parser.add_argument_group("plot arguments")
        required.add_argument("plot_type",
            type=str,
            choices=("sequence_sample",),
            help="What to plot.")
        required.add_argument("datagroup",
            type=str,
            help="Which datagroup to plot from.")
        optional = parser.add_argument_group("plot options")
        optional.add_argument("--figure-path",
            type=str,
            metavar="plot.svg",
            help="If given, plots will not be shown using tkinter (or whatever the default is) but written to this file.")
        optional.add_argument("--shuffle-buffer-size",
            type=int,
            help="How many examples to load per label from the TFRecords.")
        optional.add_argument("--num-samples",
            type=int,
            help="How many samples to include in the plots.")
        return parser

    def plot_sequence_sample(self):
        args = self.args
        datagroup = self.state["data"][args.datagroup]
        if args.shuffle_buffer_size is None:
            shuffle_buffer_size = self.experiment_config["experiment"]["shuffle_buffer_size"] // len(self.state["label_to_index"])
        else:
            shuffle_buffer_size = args.shuffle_buffer_size
        assert shuffle_buffer_size > 0
        dataset_by_label = {}
        for label, tfrecord_path in datagroup["features"].items():
            dataset, _ = system.load_features_as_dataset(
                [tfrecord_path],
                training_config={"shuffle_buffer_size": shuffle_buffer_size}
            )
            dataset_by_label[label] = np.array([example.numpy() for example, _ in dataset.take(shuffle_buffer_size)])
        visualization.plot_sequence_features_sample(dataset_by_label, figpath=args.figure_path, sample_width=args.num_samples)
        return 0

    def run(self):
        super().run()
        if self.args.plot_type == "sequence_sample":
            return self.plot_sequence_sample()
        return 1


class Kaldi(Command):

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        required = parser.add_argument_group("kaldi arguments")
        required.add_argument("wavscp", type=str, action=ExpandAbspath)
        required.add_argument("durations", type=str, action=ExpandAbspath)
        required.add_argument("utt2lang", type=str, action=ExpandAbspath)
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
        file2lang = dict(parse_kaldifile(args.utt2lang))
        utt2seg = {}
        utt2lang = {}
        for wavpath, dur in parse_kaldifile(args.durations):
            dur = float(dur)
            fileid = os.path.basename(wavpath).split(".wav")[0]
            for utt_num in range(int((dur + args.offset) / args.window_len)):
                start = utt_num * args.offset
                uttid = "{}_{}".format(fileid, utt_num)
                utt2seg[uttid] = (fileid, start, start + args.window_len)
                utt2lang[uttid] = file2lang[fileid]
        with open(os.path.join(os.path.dirname(args.wavscp), "segments"), "w") as f:
            for uttid, (fileid, start, end) in utt2seg.items():
                print(uttid, fileid, format(float(start), ".2f"), format(float(end), ".2f"), file=f)
        shutil.copyfile(args.utt2lang, args.utt2lang + ".old")
        with open(args.utt2lang, "w") as f:
            for uttid, lang in utt2lang.items():
                print(uttid, lang, file=f)

    def run(self):
        super().run()
        if self.args.task == "uniform-segment":
            return self.uniform_segment()
        return 1


command_tree = [
    (Util, [Plot]),
    (Kaldi, []),
]
