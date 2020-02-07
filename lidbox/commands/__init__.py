"""
Command definitions for all tools.
"""
import argparse
import itertools
import os

import lidbox
from . import util
from . import e2e


def create_argparser():
    """Root argparser for all lidbox commands"""
    root_parser = argparse.ArgumentParser(
        prog=lidbox.__name__,
        description=lidbox.__doc__,
    )
    subparsers = root_parser.add_subparsers(title="subcommands")
    command_tree = itertools.chain(
        util.command_tree,
        e2e.command_tree,
    )
    # Create command line options for all valid commands
    for command_group, subcommands in command_tree:
        # Add subparser for this command group
        group_argparser = command_group.create_argparser(subparsers)
        if not subcommands:
            group_argparser.set_defaults(cmd_class=command_group)
        else:
            group_subparsers = group_argparser.add_subparsers(title="subcommands", dest="subcommand")
            for subcommand in subcommands:
                subparser = subcommand.create_argparser(group_subparsers)
                # Use the class cmd for initializing a runnable command object for this subcommand
                subparser.set_defaults(cmd_class=subcommand)
    return root_parser
