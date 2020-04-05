"""
Command line entrypoint.
"""
import os
import sys
import time

import lidbox
import lidbox.cli


def main():
    profile = None
    if "--run-cProfile" in sys.argv:
        import cProfile
        import pstats
        profile_file = "cProfile.log"
        print("Running cProfile, writing output to '{}'".format(profile_file), file=sys.stderr)
        profile = cProfile.Profile()
        profile.enable()
    parser = lidbox.cli.create_argparser()
    if len(sys.argv) < 2:
        parser.error("Too few arguments, run '{} --help' for more information.".format(lidbox.__name__))
    # TODO when a subcommand is used incorrectly, get usage strings for its subparser  instead of the root parser
    args = parser.parse_args()
    tf_profiler = None
    if args.run_tf_profiler:
        from tensorflow.python.eager import profiler as tf_profiler
        tf_profile_file = os.path.abspath(os.path.join("tf_profile", str(int(time.time()))))
        print("Running TensorFlow profiler, writing output to '{}'".format(tf_profile_file), file=sys.stderr)
        tf_profiler.start()
    ret = 1
    try:
        # Initialize a Command object from the class specified in args.cmd_class and remove the class from args
        command = args.__dict__.pop("cmd_class")(args)
        ret = command.run()
    finally:
        if tf_profiler:
            tf_profiler_result = tf_profiler.stop()
            tf_profiler.save(tf_profile_file, tf_profiler_result)
        if profile:
            profile.disable()
            with open(profile_file, "w") as out_f:
                pstats.Stats(profile, stream=out_f).sort_stats("tottime").print_stats()
    if ret:
        sys.exit(ret)
