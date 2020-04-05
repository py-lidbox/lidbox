"""
Speech classification toolbox built on top of TensorFlow.
"""
import io
import logging
import os
import sys

import yaml


DEBUG = os.environ.get("LIDBOX_DEBUG") not in (None, "False", "false", "0")

logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        format="{asctime:s}.{msecs:03.0f} {levelname[0]:s} {name:s}: {message:s}")

def get_package_root():
    from . import __path__
    return os.path.abspath(os.path.dirname(__path__[0]))

def yaml_pprint(d, left_pad=0, to_string=False, **kwargs):
    assert not to_string or "file" not in kwargs, "'to_string' specified with 'file' in kwargs for yaml_pprint, writing to string stream would override the other stream"
    def _do_print():
        for line in yaml.dump(d, indent=2).splitlines():
            print(left_pad * ' ', line, sep='', **kwargs)
    if to_string:
        with io.StringIO() as sstream:
            kwargs["file"] = sstream
            _do_print()
            return sstream.getvalue()
    else:
        _do_print()

def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
