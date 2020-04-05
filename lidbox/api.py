import logging

logger = logging.getLogger("api")

import lidbox
# import lidbox.dataset



def merge_dataset_configs(config):
    num_datasets = len(config["datasets"])
    labels = sorted(set(label for dataset in config["datasets"] for label in dataset["labels"]))
    keys = sorted(set(key for dataset in config["datasets"] for key in dataset["keys"]))
    return config


def run_feature_extraction(config_file_path):
    """
    Given a path to lidbox configuration file, load the file and run its feature extraction pipeline.
    Keras models will not be loaded or trained.
    """
    logger.info("Running feature extraction using config file from %s", config_file_path)
    config = merge_dataset_configs(lidbox.load_yaml(config_file_path))
    contents_string = lidbox.yaml_pprint(config, left_pad=2, to_string=True)
    logger.debug("Config file read with contents:\n%s", contents_string)


# def load_config_file(path):
#     logger.info("Loading yaml config file '%s'", path)
#     with open(path) as f:
#         config = yaml.safe_load(f)
#     sstream = io.StringIO()
#     lidbox.yaml_pprint(config, left_pad=2, file=sstream)
#     logger.debug("Config file contents:\n%s", sstream.getvalue())
#     del sstream
#     # Merge list of dataset configs
#     labels = sorted(set(label for label in dataset["labels"] for dataset in config["datasets"]))
#     splits = {}
