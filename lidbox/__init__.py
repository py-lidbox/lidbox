"""
Toolbox containing various speech data analysis tools.
"""
import os
import yaml

def get_package_root():
    from lidbox import __path__
    return os.path.abspath(os.path.dirname(__path__[0]))

def yaml_pprint(d, **kwargs):
    print(yaml.dump(d, indent=2), **kwargs)
