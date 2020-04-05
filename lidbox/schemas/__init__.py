import io
import os

import jsonschema

import lidbox

try:
    CONFIG_FILE_SCHEMA_PATH = os.path.join(__path__[0], "config.yaml")
except Exception as e:
    print("Warning: unable to load JSON schema path, error was:")
    print(str(e))
    CONFIG_FILE_SCHEMA_PATH = None


def validate_config_file_and_get_error_string(path, verbose):
    schema = lidbox.load_yaml(CONFIG_FILE_SCHEMA_PATH)
    config = lidbox.load_yaml(path)
    error_string = ''
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as error:
        with io.StringIO() as sstream:
            print("File '{}' validation failed, error is:\n  {}".format(path, error.message), file=sstream)
            if error.context:
                print("context:", file=sstream)
                for context in error.context:
                    print(context, file=sstream)
            if error.cause:
                print("cause:\n", error.cause, file=sstream)
            if verbose:
                print("Instance was:", file=sstream)
                lidbox.yaml_pprint(error.instance, left_pad=2, file=sstream)
            error_string = sstream.getvalue()
    return error_string
