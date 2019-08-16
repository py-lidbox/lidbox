"""
Command line entrypoint.
"""
import sys

import speechbox
from speechbox.commands import create_argparser


def main():
    profile = None
    if "--run-cProfile" in sys.argv:
        import cProfile
        import pstats
        profile_file = "cProfile.log"
        print("Running cProfile, writing output to '{}'".format(profile_file), file=sys.stderr)
        profile = cProfile.Profile()
        profile.enable()
    parser = create_argparser()
    if len(sys.argv) < 2:
        parser.error("Too few arguments, run '{} --help' for more information.".format(speechbox.__name__))
    # TODO when a subcommand is used incorrectly, get usage strings for its subparser  instead of the root parser
    args = parser.parse_args()
    # Initialize a Command object from the class specified in args.cmd_class and remove the class from args
    command = args.__dict__.pop("cmd_class")(args)
    ret = command.run() or command.exit()
    if profile:
        profile.disable()
        with open(profile_file, "w") as out_f:
            pstats.Stats(profile, stream=out_f).sort_stats("tottime").print_stats()
    if ret:
        sys.exit(ret)
