import sys

from speechbox.commands import ExpandAbspath
from speechbox.commands.base import Command
import speechbox.dataset as dataset


class Parser(Command):
    tasks = ("parse",)

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("src",
            type=str,
            action=ExpandAbspath,
            help="Parse files from this directory.")
        parser.add_argument("dst",
            type=str,
            action=ExpandAbspath,
            help="Save parsed output into this directory.")
        parser.add_argument("--parse",
            choices=dataset.all_parsers,
            help="Parse from --src to --dst using the given dataset parser.")
        parser.add_argument("--resample",
            type=int,
            help="Resample all output files to the given sample frequency.")
        parser.add_argument("--limit",
            type=int,
            help="Only parse this many audio files, starting from the files with the highest |num_upvotes - num_downvotes| value.")
        parser.add_argument("--fail-early",
            action="store_true",
            default=False,
            help="Stop parsing when an error occurs instead of skipping over the error file.")
        return parser

    def parse(self):
        args = self.args
        if args.verbosity:
            print("Parsing dataset '{}'".format(args.parse))
        if not (self.args_src_ok() and self.args_dst_ok()):
            return 1
        parser_config = {
            "dataset_root": args.src,
            "output_dir": args.dst,
            "resampling_freq": args.resample,
            "output_count_limit": args.limit,
            "fail_early": args.fail_early,
        }
        parser = dataset.get_dataset_parser(args.parse, parser_config)
        num_parsed = 0
        if not args.verbosity:
            for path, output in parser.parse():
                if output is not None:
                    num_parsed += 1
        else:
            for path, output in parser.parse():
                if output is None:
                    print("Warning: failed to parse '{}'".format(path))
                    continue
                num_parsed += 1
                if args.verbosity > 1 and any(output):
                    status, out, err = output
                    msg = "Warning:"
                    if status:
                        msg += " exit code: {}".format(status)
                    if out:
                        print(msg, out)
                    if err:
                        print(msg, err, file=sys.stderr)
                if num_parsed % 1000 == 0:
                    print(num_parsed, "files parsed")
        print(num_parsed, "files parsed from '{}' to '{}'".format(args.src, args.dst))

    def run(self):
        super().run()
        return self.run_tasks()
