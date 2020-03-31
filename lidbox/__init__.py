"""
Toolbox containing various speech data analysis tools.
"""
import os
import yaml

TF_DEBUG = bool(os.environ.get("LIDBOX_TF_DEBUG", False))

def get_package_root():
    from . import __path__
    return os.path.abspath(os.path.dirname(__path__[0]))

try:
    from . import __name__ as package_name
    CONFIG_FILE_SCHEMA_PATH = os.path.join(get_package_root(), package_name, "schemas", "config.yaml")
except Exception as e:
    print("Warning: unable to load JSON schema path, error was:")
    print(str(e))
    CONFIG_FILE_SCHEMA_PATH = None

def yaml_pprint(d, left_pad=0, **kwargs):
    for line in yaml.dump(d, indent=2).splitlines():
        print(left_pad * ' ', line, sep='', **kwargs)

def parse_space_separated(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                yield l.split(' ')
