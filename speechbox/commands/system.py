import pprint

from speechbox.commands import ExpandAbspath
from speechbox.commands.base import Command
import speechbox.system as system


class System(Command):
    """CLI for IO functions in speechbox.system."""

    tasks = ("yaml_get", "get_unique_duration")

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
        return parser

    def yaml_get(self):
        args = self.args
        for infile in args.infile:
            value = system.load_yaml(infile)
            for key in args.yaml_get.split('.'):
                value = value[key]
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

    def run(self):
        return self.run_tasks()
