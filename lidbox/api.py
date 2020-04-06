import collections
import importlib
import itertools
import logging
import os
import random
import sys

logger = logging.getLogger("api")

import numpy as np
import sklearn.metrics

import lidbox
from lidbox.dataset.steps import Step
from lidbox.models.keras_utils import KerasWrapper


VALID_METADATA_FILES = {
    "utt2dur": "duration",
    "utt2duration": "duration",
    "utt2label": "label",
    "utt2lang": "label",
    "utt2path": "path",
    "utt2spk": "speaker",
}


def get_flat_dataset_config(config):
    num_datasets = len(config["datasets"])
    # Merge all labels and sort
    labels = sorted(set(label for dataset in config["datasets"] for label in dataset["labels"]))
    split2datasets = collections.defaultdict(list)
    for dataset in config["datasets"]:
        for split in dataset["splits"]:
            split = dict(split)
            meta = {VALID_METADATA_FILES[p.name]: p.path for p in os.scandir(split.pop("path")) if p.name in VALID_METADATA_FILES}
            meta["dataset"] = dataset["key"]
            meta["kwargs"] = split
            split2datasets[split.pop("key")].append(meta)
    #TODO assert amount of keys and all values of same length
    return dict(split2datasets), labels


def load_all_metadata_from_paths(split2datasets):
    split2datasets_meta = collections.OrderedDict()
    for split, datasets in split2datasets.items():
        split2datasets_meta[split] = []
        for meta in datasets:
            meta = dict(meta)
            dataset = meta.pop("dataset")
            kwargs = meta.pop("kwargs")
            # Read all meta files
            meta = {key: collections.OrderedDict(lidbox.iter_metadata_file(path, num_columns=2)) for key, path in meta.items()}
            # 'utt2path' is always present, use it to select final utterance ids
            utt_ids = list(meta["path"].keys())
            if kwargs.get("shuffle_files", False):
                random.shuffle(utt_ids)
            utt_ids = utt_ids[:kwargs.get("file_limit", -1)]
            # Filter all metadata with selected utterance ids and ensure correct order
            meta = {key: [utt2meta[utt] for utt in utt_ids] for key, utt2meta in meta.items()}
            meta["id"] = utt_ids
            meta["dataset"] = len(utt_ids) * [dataset]
            split2datasets_meta[split].append(meta)
    return split2datasets_meta


def merge_dataset_metadata(split2datasets_meta):
    split2meta = {}
    for split, datasets in split2datasets_meta.items():
        split2meta[split] = {key: data for key, data in datasets[0].items()}
        for dataset in datasets[1:]:
            for key, data in dataset.items():
                split2meta[split][key].extend(data)
    return split2meta


def load_user_script_as_module(path):
    spec = importlib.util.spec_from_file_location("lidbox.user_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_splits_from_config_file(config_file_path):
    logger.info("Using config file '%s'", config_file_path)
    config = lidbox.load_yaml(config_file_path)
    logger.debug("Reading all metadata from %d different datasets.", len(config["datasets"]))
    split2datasets, labels = get_flat_dataset_config(config)
    logger.info("Merged all metadata into %d splits, set of all labels is:\n  %s", len(split2datasets), '\n  '.join(labels))
    logger.debug("Loading metadata from all files and merging metadata of all datasets")
    split2meta = merge_dataset_metadata(load_all_metadata_from_paths(split2datasets))
    return split2meta, labels, config


def create_datasets_using_user_script(split2meta, labels, config):
    logger.info("Loading user script at '%s'.", config["user_script"])
    user_script = load_user_script_as_module(config["user_script"])
    split2ds = {}
    for split, split_meta in split2meta.items():
        steps = user_script.create_dataset(split, labels, split_meta, config)
        split2ds[split] = lidbox.dataset.from_steps(steps)
    return split2ds


def run_training(split2ds, config):
    from lidbox.dataset.steps import as_supervised
    from lidbox.models.keras_utils import best_model_checkpoint_from_config
    split_conf = config["experiment"]["data"]
    # 1. get the training and validation splits as defined by the user
    # 2. batch the datasets
    # 3. drop all dictionary keys and convert each element to (inputs, targets) pairs
    train_ds = split2ds[split_conf["train"]["split"]]
    shuffle_buffer_size = split_conf["train"].get("shuffle_buffer_size")
    if shuffle_buffer_size is not None:
        logger.info("Shuffling training dataset with buffer of size %d", shuffle_buffer_size)
        train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = (train_ds
                    .batch(split_conf["train"]["batch_size"])
                    .apply(as_supervised))
    validation_ds = (split2ds[split_conf["validation"]["split"]]
                        .batch(split_conf["validation"]["batch_size"])
                        .apply(as_supervised))
    user_script = load_user_script_as_module(config["user_script"])
    if hasattr(user_script, "train"):
        logger.info("User script has defined a 'train' function, will use it")
        history = user_script.train(train_ds, validation_ds, config)
        if history is None:
            logger.warning("Function 'train' in the user script '%s' did not return a history object", config["user_script"])
    else:
        logger.info("User script has not defined a 'train' function, will use default approach")
        keras_wrapper = KerasWrapper.from_config(config)
        logger.info("Model initialized:\n%s", str(keras_wrapper))
        best_checkpoint = best_model_checkpoint_from_config(config)
        if best_checkpoint:
            logger.info("Found existing model checkpoint '%s', loading weights from it and continuing training", best_checkpoint)
            keras_wrapper.load_weights(best_checkpoint)
        history = keras_wrapper.fit(train_ds, validation_ds, config["experiment"].get("keras_fit_kwargs", {}))
    return history


def group_chunk_predictions_by_parent_id(utt2prediction):
    """
    Group all chunks by the parent utterance id separated by '-' and take average over chunk predictions.
    """
    return [(utt, np.stack([pred for _, pred in chunk2pred]).mean(axis=0))
            for utt, chunk2pred in
            itertools.groupby(utt2prediction, key=lambda t: t[0].rsplit('-', 1)[0])]


def print_predictions(utt2prediction, labels, precision=3, **print_kwargs):
    print(*labels, **print_kwargs)
    for utt, pred in utt2prediction:
        scores_str = [np.format_float_positional(x, precision=precision) for x in pred]
        print(utt, *scores_str, **print_kwargs)


#TODO simplify and divide into manageable pieces
def evaluate_test_set(split2ds, split2meta, labels, config):
    from lidbox.dataset.steps import as_supervised, initialize
    from lidbox.models.keras_utils import best_model_checkpoint_from_config, experiment_cache_from_config
    test_conf = config["experiment"]["data"]["test"]
    test_ds = (split2ds[test_conf["split"]]
                .batch(test_conf["batch_size"])
                .apply(as_supervised))
    user_script = load_user_script_as_module(config["user_script"])
    if hasattr(user_script, "predict"):
        logger.info("User script has defined a 'predict' function, will use it")
        predictions = user_script.predict(test_ds, config)
        if predictions is None:
            logger.error("Function 'predict' in the user script '%s' did not return predictions", config["user_script"])
            return
    else:
        logger.info("User script has not defined a 'predict' function, will use default approach")
        keras_wrapper = KerasWrapper.from_config(config)
        logger.info("Model initialized:\n%s", str(keras_wrapper))
        best_checkpoint = best_model_checkpoint_from_config(config)
        logger.info("Loading weights from checkpoint file '%s'", best_checkpoint)
        keras_wrapper.load_weights(best_checkpoint)
        logger.info("Starting prediction with model '%s'", keras_wrapper.model_key)
        predictions = keras_wrapper.keras_model.predict(test_ds)
    logger.info("Model predicted results for %d inputs, now gathering all test set ids", predictions.shape[0])
    test_ids = [x["id"].decode("utf-8") for x in split2ds[test_conf["split"]].as_numpy_iterator()]
    utt2prediction = sorted(zip(test_ids, predictions), key=lambda t: t[0])
    del test_ids
    if "chunks" in config.get("pre_process", {}):
        utt2prediction = group_chunk_predictions_by_parent_id(utt2prediction)
        predictions = np.array([p for _, p in utt2prediction])
    test_meta_ds = initialize(None, labels, split2meta[test_conf["split"]])
    utt2target = [(x["id"].decode("utf-8"), x["target"]) for x in test_meta_ds.as_numpy_iterator()]
    missed_utterances = set(u for u, _ in utt2target) - set(u for u, _ in utt2prediction)
    min_score = np.amin(predictions)
    max_score = np.amax(predictions)
    if missed_utterances:
        logger.info("%d test samples had no predictions and worst-case scores %.3f will be generated for them for every label", len(missed_utterances), min_score)
        utt2prediction.extend([(utt, np.array([min_score for _ in labels])) for utt in sorted(missed_utterances)])
        predictions = np.array([p for _, p in utt2prediction])
    scores_file = os.path.join(experiment_cache_from_config(config), "predictions")
    logger.info("Writing predicted scores to '%s'", scores_file)
    with open(scores_file, "w") as scores_f:
        print_predictions(utt2prediction, labels, file=scores_f)
    metric_results = []
    true_labels_sparse = np.array([t for _, t in utt2target])
    pred_labels_sparse = np.argmax(predictions, axis=1)
    logger.info("Evaluating metrics on true labels of shape %s and predicted labels of shape %s", true_labels_sparse.shape, pred_labels_sparse.shape)
    for metric in test_conf["evaluate_metrics"]:
        result = None
        if metric["name"].endswith("average_detection_cost"):
            logger.debug("Evaluating minimum average detection cost")
            thresholds = np.linspace(min_score, max_score, metric.get("num_thresholds", 200))
            if metric["name"].startswith("sparse_"):
                cavg = lidbox.metrics.SparseAverageDetectionCost(len(labels), thresholds)
                cavg.update_state(np.expand_dims(true_labels_sparse, -1), predictions)
            else:
                cavg = lidbox.metrics.AverageDetectionCost(len(labels), thresholds)
                cavg.update_state(true_labels, predictions)
            result = float(cavg.result().numpy())
            logger.info("%s: %.6f", metric["name"], result)
        elif metric["name"].endswith("average_equal_error_rate"):
            #TODO sparse EER, generate one-hot true_labels
            logger.debug("Evaluating average equal error rate")
            eer = np.zeros(len(labels))
            for l, label in enumerate(labels):
                if label not in all_testset_labels:
                    eer[l] = 0
                    continue
                # https://stackoverflow.com/a/46026962
                fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels[:,l], predictions[:,l])
                fnr = 1 - tpr
                eer[l] = fpr[np.nanargmin(np.absolute(fnr - fpr))]
            result = {"avg": float(eer.mean()),
                      "by_label": {label: float(eer[l]) for l, label in enumerate(labels)}}
            logger.info("%s: %s", metric["name"], lidbox.yaml_pprint(result, to_string=True))
        elif metric["name"] == "average_f1_score":
            logger.debug("Evaluating average F1 score")
            f1 = sklearn.metrics.f1_score(
                    true_labels_sparse,
                    pred_labels_sparse,
                    labels=list(range(len(labels))),
                    average="weighted")
            result = {"avg": float(f1)}
            logger.info("%s: %.6f", metric["name"], f1)
        elif metric["name"] == "sklearn_classification_report":
            logger.debug("Generating full sklearn classification report")
            result = sklearn.metrics.classification_report(
                    true_labels_sparse,
                    pred_labels_sparse,
                    labels=list(range(len(labels))),
                    target_names=labels,
                    output_dict=True,
                    zero_division=0)
            logger.info("%s:\n%s", metric["name"], lidbox.yaml_pprint(result, left_pad=2, to_string=True))
        elif metric["name"] == "confusion_matrix":
            logger.debug("Generating confusion matrix")
            result = sklearn.metrics.confusion_matrix(true_labels_sparse, pred_labels_sparse)
            logger.info("%s:\n%s", metric["name"], str(result))
            result = result.tolist()
        else:
            logger.error("Cannot evaluate unknown metric '%s'", metric["name"])
        metric_results.append({"name": metric["name"], "result": result})
