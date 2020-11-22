"""
Speech classification toolbox built on top of TensorFlow.
"""
import io
import logging
import os
import random
import sys


random.seed(int(os.environ.get("LIDBOX_RANDOM_SEED", "42")))

DEBUG = os.environ.get("LIDBOX_DEBUG") not in (None, "False", "false", "0")

TF_LOG_FORMAT = logging.Formatter(
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        fmt="{asctime:s}.{msecs:03.0f} {levelname[0]:s} {name:s}: {message:s}")

def reset_global_loglevel(level):
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)
    # info and debug to stdout, warn and higher to stderr
    info_logger = logging.StreamHandler(sys.stdout)
    info_logger.setFormatter(TF_LOG_FORMAT)
    info_logger.addFilter(lambda record: record.levelno <= logging.INFO)
    root.addHandler(info_logger)
    warn_logger = logging.StreamHandler(sys.stderr)
    warn_logger.setFormatter(TF_LOG_FORMAT)
    warn_logger.addFilter(lambda record: record.levelno > logging.INFO)
    root.addHandler(warn_logger)

reset_global_loglevel(logging.DEBUG if DEBUG else logging.INFO)

def get_package_root():
    from . import __path__
    return os.path.abspath(os.path.dirname(__path__[0]))

def iter_metadata_file(path, num_columns):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                yield l.split(' ', num_columns)[:num_columns]
