import argparse
import pprint

import speechbox

def create_argparser():
    parser = argparse.ArgumentParser(prog=speechbox.__name__)
    subparsers = parser.add_subparsers(
        title="tools",
        description="subcommands for different tasks",
    )
    for cmd in VALID_COMMANDS:
        # Create command line options for this command
        subparser = cmd.create_argparser(subparsers)
        # Set entry point for running the command
        subparser.set_defaults(do_cmd=cmd.do_command)
    return parser

class Command:
    """Base command that only prints its arguments to stdout."""
    @classmethod
    def create_argparser(cls, subparsers):
        parser = subparsers.add_parser(cls.__name__.lower(), description=cls.__doc__)
        parser.add_argument("--verbosity", "-v",
            action="count",
            default=0,
            help="increase verbosity of output to stdout"
        )
        return parser
    @classmethod
    def do_command(cls, args):
        if args.verbosity > 1:
            print("Running '{}' with arguments:".format(cls.__name__.lower()))
            pprint.pprint(vars(args))

class Preprocess(Command):
    """Feature extraction and dataset analysis."""
    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("-p")
        return parser
    @classmethod
    def do_command(cls, args):
        super().do_command(args)

class Train(Command):
    """Model training."""
    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("-t")
        return parser
    @classmethod
    def do_command(cls, args):
        super().do_command(args)

class Evaluate(Command):
    """Prediction and evaluation using trained models."""
    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("-e")
        return parser
    @classmethod
    def do_command(cls, args):
        super().do_command(args)

VALID_COMMANDS = (
    Preprocess,
    Train,
    Evaluate,
)
