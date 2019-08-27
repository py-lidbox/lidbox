import collections

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
        error_msg = "No SpeechDatasetWalker class defined for dataset '{}'".format(dataset)
        raise UnknownDatasetException(error_msg)
    return all_walkers[dataset]

def group_paths_by_speaker(paths, dataset):
    parse_speaker_id = get_dataset_walker_cls(dataset).parse_speaker_id
    groups = collections.defaultdict(list)
    for path in paths:
        groups[parse_speaker_id(path)].append(path)
    return groups
