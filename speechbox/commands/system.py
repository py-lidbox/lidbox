import pprint
import time
import sys

from speechbox.commands import ExpandAbspath
from speechbox.commands.base import Command
import speechbox.system as system
import speechbox.visualization as visualization


class System(Command):
    """CLI for IO functions in speechbox.system."""

    tasks = (
        "yaml_get",
        "get_unique_duration",
        "plot_melspectrogram",
        "watch",
    )

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("infile",
            type=str,
            nargs="*",
            help="Path to one or more input files to apply the command for")
        parser.add_argument("--yaml-get",
            type=str,
            help="Parse yaml files and dump contents of given yaml key to stdout.")
        parser.add_argument("--get-unique-duration",
            action="store_true",
            help="Group all files by MD5 sums and compute total duration for all unique, valid audio files. Files that cannot be opened as audio files are ignored. If there are duplicate files by MD5 sum, the first file is chosen in the same order as it was given as argument to this command.")
        parser.add_argument("--plot-melspectrogram",
            action="store_true",
            help="Plot mel spectrograms for all given files. Blocks control until all figures are manually closed.")
        parser.add_argument("--watch",
            action="store_true")
        parser.add_argument("--transform",
            type=str,
            action=ExpandAbspath,
            help="Apply transformations from this experiment config file  as they would be performed before feature extraction. Then play audio and visualize spectrograms.")
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


    def run(self):
        return self.run_tasks()
