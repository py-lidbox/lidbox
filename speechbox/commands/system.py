import pprint

from speechbox.commands import ExpandAbspath
from speechbox.commands.base import Command
import speechbox.system as system


class System(Command):
    """CLI for IO functions in speechbox.system."""

    tasks = ("yaml_get",)

    @classmethod
    def create_argparser(cls, subparsers):
        parser = super().create_argparser(subparsers)
        parser.add_argument("src",
            type=str,
            action=ExpandAbspath,
            help="Path to a file, depends on task.")
        parser.add_argument("--yaml-get",
            type=str,
            help="Parse yaml file at src and dump contents of given yaml key to stdout.")
        return parser

    def yaml_get(self):
        args = self.args
        value = system.load_yaml(args.src)
        for key in args.yaml_get.split('.'):
            value = value[key]
        if type(value) is list:
            value_str = '\n'.join(value)
        elif type(value) is dict:
            value_str = pprint.pformat(value, indent=1)
        else:
            value_str = str(value)
        print(value_str)

    def run(self):
        return self.run_tasks()
