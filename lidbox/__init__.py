"""
Toolbox containing various speech data analysis tools.
"""
import os
import yaml

TF_DEBUG = bool(os.environ.get("LIDBOX_TF_DEBUG", False))

def get_package_root():
    from . import __path__
    return os.path.abspath(os.path.dirname(__path__[0]))

def yaml_pprint(d, **kwargs):
    print(yaml.dump(d, indent=2), **kwargs)

def parse_space_separated(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                yield l.split(' ')
