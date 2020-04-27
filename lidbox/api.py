"""
High-level interface for interacting with lidbox.
"""
import collections
import importlib
import itertools
import json
import logging
import os
import random
import sys

logger = logging.getLogger("api")

import numpy as np
import sklearn.metrics

import lidbox
import lidbox.dataset.steps
import lidbox.models.keras_utils
from lidbox.dataset.steps import Step
from lidbox.models.keras_utils import KerasWrapper


# When scanning a datadir for valid metadata files, these filenames will be accepted, all others are ignored
VALID_METADATA_FILES = {
    "utt2dur": "duration",
    "utt2duration": "duration",
    "utt2feat.scp": "kaldi_ark_key",
    "utt2label": "label",
    "utt2lang": "label",
    "utt2path": "path",
    "utt2spk": "speaker",
    "utt2transcript": "transcript",
    "transcript.ctm": "transcript",
}


def create_datasets(split2meta, labels, config):
    from lidbox.dataset import from_steps
    create_dataset = None
    modify_steps = None
    if "user_script" in config:
        user_script = load_user_script_as_module(config["user_script"])
        create_dataset = getattr(user_script, "create_dataset", None)
        modify_steps = getattr(user_script, "modify_steps", None)
    if create_dataset is None:
        from lidbox.dataset.pipelines import create_dataset
    else:
        logger.info("User has defined a 'create_dataset' function, will use it to create dataset steps")
    if modify_steps is None:
        modify_steps = lambda steps, *args: steps
    else:
        logger.info("User has defined a 'modify_steps' function, the created dataset will be given to the function for modification")
    split2ds = {}
    for split, split_meta in split2meta.items():
        logger.info("Creating dataset iterator for split '%s' with metadata containing %d keys", split, len(split_meta))
        if "pre_initialize" in config:
            logger.info("'pre_initialize' defined in config, updating metadata before creating dataset iterator.")
            from lidbox.dataset.steps import pre_initialize
            split_meta = pre_initialize(split_meta, config["pre_initialize"], labels)
        args = split, labels, split_meta, config
        steps = create_dataset(*args)
        steps = modify_steps(steps, *args)
        split2ds[split] = from_steps(steps)
    return split2ds


def get_flat_dataset_config(config):
    num_datasets = len(config["datasets"])
    # Merge all labels and sort
    labels = sorted(set(label for dataset in config["datasets"] for label in dataset["labels"]))
    split2datasets = collections.defaultdict(list)
    for dataset in config["datasets"]:
        for split in dataset["splits"]:
            split = dict(split)
            logger.info("Scanning dataset '%s' split '%s' for valid metadata files", dataset["key"], split["key"])
            meta = {VALID_METADATA_FILES[p.name]: p for p in os.scandir(split.pop("path")) if p.name in VALID_METADATA_FILES}
            logger.info("Found %d valid metadata files:\n  %s", len(meta), '\n  '.join(p.path for p in meta.values()))
            enabled_datafiles = set(split.pop("datafiles", []))
            if enabled_datafiles:
                logger.info("Key 'datafiles' given for split '%s' and contains %d datafiles, filtering metadata files by given list", split["key"], len(enabled_datafiles))
                meta = {k: v for k, v in meta.items() if v.name in enabled_datafiles}
            else:
                logger.info("Key 'datafiles' not given for split '%s', assuming all valid metadata files should be used", split["key"])
            meta = {k: v.path for k, v in meta.items()}
            logger.info("Using %d metadata files:\n  %s", len(meta), '\n  '.join(meta.values()))
            meta["dataset"] = dataset["key"]
            meta["kwargs"] = split
            split2datasets[split.pop("key")].append(meta)
    return dict(split2datasets), labels


def load_all_metadata_from_paths(split2datasets):
    split2datasets_meta = collections.OrderedDict()
    for split, datasets in split2datasets.items():
        split2datasets_meta[split] = []
        for meta in datasets:
            meta = dict(meta)
            dataset_key = meta.pop("dataset")
            kwargs = meta.pop("kwargs")
            logger.info("Loading all metadata file contents for dataset '%s' split '%s'", dataset_key, split)
            # Read all meta files
            meta = {key: collections.OrderedDict(lidbox.iter_metadata_file(path, num_columns=2)) for key, path in meta.items()}
            logger.info("Amount of contents per file:\n  %s", '\n  '.join("{}: {}".format(key, len(val)) for key, val in meta.items()))
            first_meta_length = len(list(meta.values())[0])
            if not all(len(meta_list) == first_meta_length for meta_list in meta.values()):
                logger.error("All metadata files must contain exactly the same amount of unique utterance ids")
                return
            # 'utt2path' is always present, use it to select final utterance ids
            utt_ids = list(meta["path"].keys())
            if kwargs.get("shuffle_files", False):
                logger.info("'shuffle_files' given for dataset '%s' split '%s', shuffling all its utterance ids", dataset_key, split)
                random.shuffle(utt_ids)
            file_limit = kwargs.get("file_limit")
            utt_ids = utt_ids[:file_limit]
            logger.info("After applying file_limit %s, amount of final utterance ids that will be used is %d", file_limit, len(utt_ids))
            # Filter all metadata with selected utterance ids to ensure correct order of metadata
            # This step is very important in order to not have samples with wrong metadata
            meta = {key: [utt2meta[utt] for utt in utt_ids] for key, utt2meta in meta.items()}
            meta["id"] = utt_ids
            meta["dataset"] = len(utt_ids) * [dataset_key]
            split2datasets_meta[split].append(meta)
            logger.info("Dataset '%s' split '%s' done, all its elements will have keys:\n  %s", dataset_key, split, '\n  '.join(meta.keys()))
    return split2datasets_meta


def merge_dataset_metadata(split2datasets_meta):
    split2meta = {}
    for split, datasets in split2datasets_meta.items():
        split2meta[split] = {meta_key: meta for meta_key, meta in datasets[0].items()}
        for dataset in datasets[1:]:
            for meta_key, meta in dataset.items():
                split2meta[split][meta_key].extend(meta)
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
    logger.info("Reading all metadata from %d different datasets.", len(config["datasets"]))
    split2datasets, labels = get_flat_dataset_config(config)
    logger.info("Merged all metadata into %d splits, set of all labels is:\n  %s", len(split2datasets), '\n  '.join(labels))
    logger.info("Loading metadata from all files and merging metadata of all datasets")
    split2meta = merge_dataset_metadata(load_all_metadata_from_paths(split2datasets))
    return split2meta, labels, config


def run_training(split2ds, config):
    from lidbox.models.keras_utils import best_model_checkpoint_from_config
    split_conf = config["experiment"]["data"]
    # 1. get the training and validation splits as defined by the user
    # 2. drop all dictionary keys and convert each element to (inputs, targets) pairs
    # 3. batch the datasets
    train_ds = (split2ds[split_conf["train"]["split"]]
            .apply(lidbox.dataset.steps.as_supervised)
            .batch(split_conf["train"]["batch_size"]))
    validation_ds = (split2ds[split_conf["validation"]["split"]]
            .apply(lidbox.dataset.steps.as_supervised)
            .batch(split_conf["validation"]["batch_size"]))
    #TODO split
    history = None
    if "user_script" in config:
        user_script = load_user_script_as_module(config["user_script"])
        if hasattr(user_script, "train"):
            logger.info("User script has defined a 'train' function, will use it")
            history = user_script.train(train_ds, validation_ds, config)
            if history is None:
                logger.warning("Function 'train' in the user script '%s' did not return a history object", config["user_script"])
                history = []
    if history is None:
        logger.info("User script has not defined a 'train' function, will use default approach")
        keras_wrapper = KerasWrapper.from_config(config)
        logger.info("Model initialized:\n%s", str(keras_wrapper))
        best_checkpoint = lidbox.models.keras_utils.best_model_checkpoint_from_config(config)
        if best_checkpoint:
            logger.info("Found existing model checkpoint '%s', loading weights from it and continuing training", best_checkpoint)
            keras_wrapper.load_weights(best_checkpoint)
        history = keras_wrapper.fit(train_ds, validation_ds, config["experiment"].get("keras_fit_kwargs", {}))
    return history


def group_chunk_predictions_by_parent_id(utt2prediction):
    """
    Group all chunks by the parent utterance id separated by '-' and take average over chunk predictions.
    """
    get_parent_id = lambda t: t[0].rsplit('-', 1)[0]
    return [(utt, np.stack([pred for _, pred in chunk2pred]).mean(axis=0))
            for utt, chunk2pred in
            itertools.groupby(sorted(utt2prediction, key=get_parent_id), key=get_parent_id)]


def print_predictions(utt2prediction, labels, precision=3, **print_kwargs):
    print(*labels, **print_kwargs)
    for utt, pred in utt2prediction:
        scores_str = [np.format_float_positional(x, precision=precision) for x in pred]
        print(utt, *scores_str, **print_kwargs)


def format_confusion_matrix(cm, labels):
    assert cm.shape[0] == cm.shape[1] == len(labels), "invalid confusion matrix and/or labels"
    label_format = "{{:{:d}s}}".format(max(len(l) for l in labels))
    labels_padded = [label_format.format(l) for l in labels]
    num_pred_labels = cm.sum(axis=0)
    num_true_labels = cm.sum(axis=1)
    cm_lines = str(cm).splitlines()
    cm_lines = [label + " " + cm_line + " " + str(num_true)
                for label, cm_line, num_true in zip(labels_padded, cm_lines, num_true_labels)]
    cm_lines = [label_format.format('') + ' '.join(labels)] + cm_lines
    cm_lines.append(label_format.format('') + ' '.join(str(n) for n in num_pred_labels))
    return '\n'.join(cm_lines)


def zip_and_sort_utt2vector(ds, vectors):
    def _id_only(x):
        return x["id"]
    #TODO it's unnecessary to evaluate the full pipeline (e.g. extract features twice)
    ids = [uttid.decode("utf-8") for uttid in ds.map(_id_only).as_numpy_iterator()]
    if vectors.shape[0] != len(ids):
        logger.error("Cannot combine %d ids with %d vectors", len(ids), vectors.shape[0])
        return
    return sorted(zip(ids, vectors), key=lambda t: t[0])


def extract_embeddings(dataset, config, batch_size=128):
    keras_wrapper = KerasWrapper.from_config(config)
    logger.info("Model initialized:\n%s", str(keras_wrapper))
    best_checkpoint = lidbox.models.keras_utils.best_model_checkpoint_from_config(config)
    logger.info("Loading weights from checkpoint file '%s'", best_checkpoint)
    keras_wrapper.load_weights(best_checkpoint)
    keras_wrapper.to_embedding_extractor()
    logger.info("Extracting embeddings with model '%s' in batches of %d", keras_wrapper.model_key, batch_size)
    embeddings = keras_wrapper.keras_model.predict(
            dataset.batch(batch_size).apply(lidbox.dataset.steps.as_supervised))
    logger.info("Model returned embeddings of shape %s, combining vectors with utterance ids", repr(embeddings.shape))
    return zip_and_sort_utt2vector(dataset, embeddings)


#TODO simplify and divide into manageable pieces
#TODO check user script before calling this
def evaluate_test_set(split2ds, split2meta, labels, config):
    test_conf = config["experiment"]["data"]["test"]
    test_ds = (split2ds[test_conf["split"]]
                .batch(test_conf["batch_size"])
                .apply(lidbox.dataset.steps.as_supervised))
    predictions = None
    if "user_script" in config:
        user_script = load_user_script_as_module(config["user_script"])
        if hasattr(user_script, "predict"):
            logger.info("User script has defined a 'predict' function, will use it")
            predictions = user_script.predict(test_ds, config)
            if predictions is None:
                logger.error("Function 'predict' in the user script '%s' did not return predictions", config["user_script"])
                return
    if predictions is None:
        logger.info("User script has not defined a 'predict' function, will use default approach")
        keras_wrapper = KerasWrapper.from_config(config)
        logger.info("Model initialized:\n%s", str(keras_wrapper))
        best_checkpoint = lidbox.models.keras_utils.best_model_checkpoint_from_config(config)
        logger.info("Loading weights from checkpoint file '%s'", best_checkpoint)
        keras_wrapper.load_weights(best_checkpoint)
        logger.info("Starting prediction with model '%s'", keras_wrapper.model_key)
        predictions = keras_wrapper.keras_model.predict(test_ds)
    logger.info("Model returned predictions of shape %s, now combining predictions with correct utterance ids.", repr(predictions.shape))
    utt2prediction = zip_and_sort_utt2vector(split2ds[test_conf["split"]], predictions)
    if predictions.shape[1] > len(labels):
        logger.warning(
                "The model predicted %d labels when %d correct labels were expected. All predictions are sliced up to index %d.",
                predictions.shape[1], len(labels), len(labels))
        utt2prediction = [(u, p[:len(labels)]) for u, p in utt2prediction]
        predictions = np.stack([p for _, p in utt2prediction])
    has_chunks = False
    if "chunks" in config.get("pre_process", {}):
        logger.info("Original signals were divided into chunks, merging chunk scores by averaging")
        has_chunks = True
    if "chunks" in config.get("post_process", {}):
        logger.info("Extracted features were divided into chunks, merging chunk scores by averaging")
        has_chunks = True
    if has_chunks:
        utt2prediction = group_chunk_predictions_by_parent_id(utt2prediction)
        predictions = np.stack([p for _, p in utt2prediction])
    # Collect targets from the test set iterator
    test_meta_ds = lidbox.dataset.steps.initialize(None, labels, split2meta[test_conf["split"]])
    utt2target = {x["id"].decode("utf-8"): x["target"] for x in test_meta_ds.as_numpy_iterator()}
    missed_utterances = set(utt2target.keys()) - set(u for u, _ in utt2prediction)
    min_score = np.amin(predictions)
    max_score = np.amax(predictions)
    if missed_utterances:
        logger.info("%d test samples had no predictions and worst-case scores %.3f will be generated for them for every label", len(missed_utterances), min_score)
        utt2prediction.extend([(utt, np.array(len(labels) * [min_score])) for utt in sorted(missed_utterances)])
    scores_file = os.path.join(
            lidbox.models.keras_utils.experiment_cache_from_config(config),
            "predictions",
            "scores")
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    logger.info("Writing predicted scores of shape %s to '%s'", predictions.shape, scores_file)
    if os.path.exists(scores_file):
        logger.warning("Overwriting existing '%s'", scores_file)
    with open(scores_file, "w") as scores_f:
        print_predictions(utt2prediction, labels, file=scores_f)
    metric_results = []
    # Ensure true labels are always in the same order as in predictions
    predictions = np.stack([p for _, p in utt2prediction])
    true_labels_sparse = np.array([utt2target[u] for u, _ in utt2prediction])
    pred_labels_sparse = np.argmax(predictions, axis=1)
    logger.info("Evaluating metrics on true labels of shape %s and predicted labels of shape %s", true_labels_sparse.shape, pred_labels_sparse.shape)
    for metric in test_conf["evaluate_metrics"]:
        result = None
        if metric["name"].endswith("average_detection_cost"):
            logger.info("Evaluating minimum average detection cost")
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
            logger.info("Evaluating average equal error rate")
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
            logger.info("Evaluating average F1 score")
            f1 = sklearn.metrics.f1_score(
                    true_labels_sparse,
                    pred_labels_sparse,
                    labels=list(range(len(labels))),
                    average="weighted")
            result = {"avg": float(f1)}
            logger.info("%s: %.6f", metric["name"], f1)
        elif metric["name"] == "sklearn_classification_report":
            logger.info("Generating full sklearn classification report")
            result = sklearn.metrics.classification_report(
                    true_labels_sparse,
                    pred_labels_sparse,
                    labels=list(range(len(labels))),
                    target_names=labels,
                    output_dict=True,
                    zero_division=0)
            logger.info("%s:\n%s", metric["name"], lidbox.yaml_pprint(result, left_pad=2, to_string=True))
        elif metric["name"] == "confusion_matrix":
            logger.info("Generating confusion matrix")
            result = sklearn.metrics.confusion_matrix(true_labels_sparse, pred_labels_sparse)
            logger.info("%s:\n%s", metric["name"], format_confusion_matrix(result, labels))
            result = result.tolist()
        else:
            logger.error("Cannot evaluate unknown metric '%s'", metric["name"])
        metric_results.append({"name": metric["name"], "result": result})
    return metric_results


def write_metrics(metrics, config):
    metrics_file = os.path.join(
            lidbox.models.keras_utils.experiment_cache_from_config(config),
            "predictions",
            "metrics.json")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    logger.info("Writing evaluated metrics to '%s'", metrics_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
