import sys

import speechbox
from speechbox import commands

def main():
    parser = commands.create_argparser()
    if len(sys.argv) < 2:
        parser.error("Too few arguments, run '{} --help' for more information.".format(speechbox.__name__))
    # TODO when a subcommand is used incorrectly, get usage strings for its subparser  instead of the root parser
    args = parser.parse_args()
    do_cmd = args.__dict__.pop("do_cmd")
    do_cmd(args)
