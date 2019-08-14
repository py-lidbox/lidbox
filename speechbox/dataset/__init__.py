class UnknownDatasetException(Exception): pass
class DatasetRecursionError(RecursionError): pass

from speechbox.dataset.walkers import all_walkers
from speechbox.dataset.parsers import all_parsers

all_datasets = (
    tuple(all_walkers.keys()) +
    tuple(all_parsers.keys())
)
all_split_types = (
    "by-file", # Split by file paths
    "by-speaker", # Use a dataset walker to parse speaker ids for each file to determine speaker
    "parse-pre-defined", # Parse a pre-defined split using a dataset walker
)

def get_dataset_parser(dataset, config=None):
    if config is None:
        config = {}
    if dataset not in all_parsers:
        raise UnknownDatasetException(str(dataset))
    return all_parsers[dataset](**config)

def get_dataset_walker(dataset, config=None):
    if config is None:
        config = {}
    return get_dataset_walker_cls(dataset)(**config)

def get_dataset_walker_cls(dataset):
    if dataset not in all_walkers:
        error_msg = "'{}' has no SpeechDatasetWalker defined".format(dataset)
        raise UnknownDatasetException(error_msg)
    return all_walkers[dataset]
