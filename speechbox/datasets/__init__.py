class UnknownDatasetException(Exception): pass
class DatasetRecursionError(RecursionError): pass

from speechbox.datasets.walkers import all_walkers
from speechbox.datasets.parsers import all_parsers

all_datasets = (
    tuple(all_walkers.keys()) +
    tuple(all_parsers.keys())
)

def get_dataset_parser(dataset, config):
    if dataset not in all_parsers:
        raise UnknownDatasetException(str(dataset))
    return all_parsers[dataset](**config)

def get_dataset_walker(dataset, config):
    #FIXME hack for mgb3
    if "test_dir" in config and config["test_dir"] == config["dataset_root"]:
        dataset = dataset + "-testset"
    if dataset not in all_walkers:
        raise UnknownDatasetException(str(dataset))
    walker = all_walkers[dataset](**config)
    assert len(walker.language_definitions) == config["num_labels"], "Unexpected amount of language labels, {}, in dataset walker '{!r}'.\nThe dataset config file defined num_labels as {}".format(len(walker.language_definitions), walker, config["num_labels"])
    return walker
