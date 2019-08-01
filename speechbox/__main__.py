"""
Command line entrypoint.
"""
import sys

import speechbox
from speechbox import commands

def main():
    parser = commands.create_argparser()
    if len(sys.argv) < 2:
        parser.error("Too few arguments, run '{} --help' for more information.".format(speechbox.__name__))
    # TODO when a subcommand is used incorrectly, get usage strings for its subparser  instead of the root parser
    args = parser.parse_args()
    # Initialize a Command object from the class specified in args.cmd_class and remove the class from args
    command = args.__dict__.pop("cmd_class")(args)
    sys.exit(command.run() or 0)
