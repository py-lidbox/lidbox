"""
Command definitions for all tools.
"""
import argparse
import os

import speechbox


def create_argparser():
    parser = argparse.ArgumentParser(prog=speechbox.__name__, description=speechbox.__doc__)
    subparsers = parser.add_subparsers(
        title="tools",
        description="subcommands for different tasks",
    )
    # Create command line options for all valid commands
    for cmd in all_commands:
        # Add subparser for this subcommand
        subparser = cmd.create_argparser(subparsers)
        # Use the class cmd for initializing a runnable command object for this subcommand
        subparser.set_defaults(cmd_class=cmd)
    return parser


class ExpandAbspath(argparse.Action):
    """Simple argparse action to expand path arguments to full paths using os.path.abspath."""
    def __call__(self, parser, namespace, path, *args, **kwargs):
        setattr(namespace, self.dest, os.path.abspath(path))


from speechbox.commands.dataset import Dataset
from speechbox.commands.model import Model
from speechbox.commands.preprocess import Preprocess
from speechbox.commands.parser import Parser

all_commands = (
    Dataset,
    Model,
    Preprocess,
    Parser,
)
