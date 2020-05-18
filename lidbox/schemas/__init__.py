import io
import os

import jsonschema

import lidbox

try:
    SCHEMA_PATHS = {
        key: os.path.join(__path__[0], filename)
        for key, filename in
        [("schema", "config.yaml"),
         ("definitions", "definitions.yaml")]}
except Exception as e:
    print("Warning: unable to load JSON schema path, error was:")
    print(str(e))
    SCHEMA_PATHS = {}


def validate_config_dict_and_get_error_string(config, verbose=1):
    schema = lidbox.load_yaml(SCHEMA_PATHS["schema"])
    if "definitions" in schema:
        print("WARNING: 'definitions' already in defined in '{}', they will be overwritten with contents from '{}'".format(SCHEMA_PATHS["schema"], SCHEMA_PATHS["definitions"]))
    schema["definitions"] = lidbox.load_yaml(SCHEMA_PATHS["definitions"])
    error_string = ''
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as error:
        with io.StringIO() as sstream:
            print("Validation failed, error is:\n  {}".format(error.message), file=sstream)
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


def validate_config_file_and_get_error_string(path, verbose):
    config = lidbox.load_yaml(path)
    return validate_config_dict_and_get_error_string(config, verbose)
