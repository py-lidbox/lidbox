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

logger = logging.getLogger(__name__)

import numpy as np

import lidbox
import lidbox.dataset.steps
import lidbox.models.keras_utils
from lidbox import load_yaml, yaml_pprint
from lidbox.dataset.steps import Step
from lidbox.models.keras_utils import KerasWrapper
from lidbox.dataset.tf_utils import make_label2onehot


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
    config = load_yaml(config_file_path)
    logger.info("Reading all metadata from %d different datasets.", len(config["datasets"]))
    split2datasets, labels = get_flat_dataset_config(config)
    logger.info("Merged all metadata into %d splits, amount of all labels %s, set of all labels:\n  %s", len(split2datasets), len(labels), '\n  '.join(labels))
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


def merge_chunk_predictions(utt2prediction, merge_fn=np.mean):
    """
    Group all chunks by the parent utterance id separated by '-' and take average over chunk predictions.
    """
    get_parent_id = lambda t: t[0].rsplit('-', 1)[0]
    return [(utt, merge_fn(np.stack([pred for _, pred in chunk2pred]), axis=0))
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
    str_max_len = np.iinfo(cm.dtype).max
    cm_lines = np.array2string(cm, threshold=str_max_len, max_line_width=str_max_len).splitlines()
    cm_lines = [label + " " + cm_line + " " + str(num_true)
                for label, cm_line, num_true in zip(labels_padded, cm_lines, num_true_labels)]
    cm_lines = [label_format.format('') + ' '.join(labels)] + cm_lines
    cm_lines.append(label_format.format('') + ' '.join(str(n) for n in num_pred_labels))
    return '\n'.join(cm_lines)


def zip_utt2vector(ds, vectors):
    def _id_only(x):
        return x["id"]
    #TODO it's unnecessary to evaluate the full pipeline (e.g. extract features twice)
    ids = [uttid.decode("utf-8") for uttid in ds.map(_id_only).as_numpy_iterator()]
    if vectors.shape[0] != len(ids):
        logger.error("Cannot combine %d ids with %d vectors", len(ids), vectors.shape[0])
        return []
    return zip(ids, vectors)


def collect_targets(labels, meta):
    meta_ds = lidbox.dataset.steps.initialize(labels, meta)
    for x in meta_ds.as_numpy_iterator():
        yield x["id"].decode("utf-8"), x["target"]


def extract_embeddings_as_numpy_data(split2ds, labels):
    import tensorflow as tf
    # We assume the label2target mapping used during training was from the same labels list
    label2target, _ = make_label2onehot(labels)
    target2label = {label2target.lookup(tf.convert_to_tensor(l)).numpy(): l for l in labels}

    def _assert_valid_targets(x):
        tf.debugging.assert_equal(
                label2target.lookup(x["label"]),
                x["target"],
                message="Sample had mismatching labels and targets")
        return x

    split2numpy_ds = {s: {} for s in split2ds}
    for split, ds in split2ds.items():
        logger.info("Extracting embeddings from split '%s' and collecting them to numpy arrays", split)
        batch_X, batch_y, batch_ids = [], [], []
        for x in ds.map(_assert_valid_targets).as_numpy_iterator():
            batch_X.append(x["embedding"])
            batch_y.append(x["target"])
            batch_ids.append(x["id"])
        split2numpy_ds[split] = {
                "X": np.concatenate(batch_X),
                "y": np.concatenate(batch_y),
                "ids": np.concatenate(batch_ids).astype(str)}
    return split2numpy_ds, target2label


def fit_embedding_classifier(split2ds, split2meta, labels, config):
    import lidbox.embeddings.sklearn_utils

    split2numpy_ds, target2label = extract_embeddings_as_numpy_data(split2ds, labels)
    all_labels = set(labels)

    train_data = split2numpy_ds[config["sklearn_experiment"]["data"]["train"]["split"]]
    test_data = split2numpy_ds[config["sklearn_experiment"]["data"]["test"]["split"]]

    model_key = config["sklearn_experiment"]["model"]["key"]
    model_kwargs = config["sklearn_experiment"]["model"].get("kwargs", {})
    if model_key == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        Classifier = GaussianNB
    else:
        logger.error("Unknown model key '%s' for training embeddings.", model_key)
        return []

    sklearn_objs = lidbox.embeddings.sklearn_utils.fit_classifier(
            train_data, test_data, labels, config, target2label, Classifier, **model_kwargs)
    joblib_dir = lidbox.embeddings.sklearn_utils.pipeline_to_disk(config, sklearn_objs)
    logger.info("Wrote trained classification pipeline to '%s'", joblib_dir)


def predict_with_embedding_classifier(split2ds, split2meta, labels, config, data_conf):
    import lidbox.embeddings.sklearn_utils

    all_labels = set(labels)
    split_key = data_conf["split"]
    split2numpy_ds, target2label = extract_embeddings_as_numpy_data(
            {split_key: split2ds[split_key]}, labels)
    predict_input = split2numpy_ds[split_key]

    sklearn_objs = lidbox.embeddings.sklearn_utils.pipeline_from_disk(config)
    predictions = lidbox.embeddings.sklearn_utils.predict_with_trained_classifier(
            predict_input, config, target2label, sklearn_objs)

    utt2prediction = list(zip(predict_input["ids"], predictions))
    utt2prediction = unchunk_predictions(utt2prediction, config)

    label2target = {l: t for t, l in target2label.items()}
    utt2target = {
            u: label2target[l]
            for u, l in zip(split2meta[split_key]["id"], split2meta[split_key]["label"])
            if l in all_labels}
    utt2prediction = generate_worst_case_predictions_for_missed_utterances(
            utt2prediction, utt2target, labels)
    assert len(utt2prediction) == len(utt2target), "invalid amount of predictions {} when utt2target has {} keys".format(len(utt2prediction), len(utt2target))
    return utt2prediction, utt2target


def unchunk_predictions(utt2prediction, config):
    if "chunks" in config.get("post_process", {}):
        logger.info("Extracted features were divided into chunks, merging feature chunk scores by averaging")
        utt2prediction = merge_chunk_predictions(utt2prediction)
    if "chunks" in config.get("pre_process", {}):
        logger.info("Original signals were divided into chunks, merging signal chunk scores by averaging")
        utt2prediction = merge_chunk_predictions(utt2prediction)
    return utt2prediction


def generate_worst_case_predictions_for_missed_utterances(utt2prediction, utt2target, labels):
    missed_utterances = set(utt2target.keys()) - set(u for u, _ in utt2prediction)
    if missed_utterances:
        logger.info("%d test samples had no predictions, generating worst case scores for all missing predictions.", len(missed_utterances))
        logger.debug("missed utterances:\n  %s", "\n  ".join(str(u) for u in missed_utterances))
        predictions = np.stack([p for _, p in utt2prediction])
        min_score = np.amin(predictions)
        logger.info("Worst-case score: %.3f", min_score)
        utt2prediction.extend([(utt, np.array(len(labels) * [min_score])) for utt in sorted(missed_utterances)])
    return utt2prediction


def predict_with_keras_model(split2ds, split2meta, labels, config, data_conf):
    ds = (split2ds[data_conf["split"]]
                .batch(data_conf["batch_size"])
                .apply(lidbox.dataset.steps.as_supervised))

    keras_wrapper = KerasWrapper.from_config(config)
    logger.info("Model initialized:\n%s", str(keras_wrapper))
    best_checkpoint = lidbox.models.keras_utils.best_model_checkpoint_from_config(config)
    logger.info("Loading weights from checkpoint file '%s'", best_checkpoint)
    keras_wrapper.load_weights(best_checkpoint)
    logger.info("Starting prediction with model '%s'", keras_wrapper.model_key)
    predictions = keras_wrapper.keras_model.predict(ds)

    logger.info("Model returned predictions of shape %s.", repr(predictions.shape))
    if predictions.shape[1] > len(labels):
        logger.warning(
                "Predictions contain %d labels, but %d correct labels were expected. All predictions are sliced up to index %d.",
                predictions.shape[1], len(labels), len(labels))
        utt2prediction = [(u, p[:len(labels)]) for u, p in utt2prediction]

    logger.info("Combining predictions with utterance ids.")
    utt2prediction = sorted(
            zip_utt2vector(split2ds[data_conf["split"]], predictions),
            key=lambda t: t[0])
    utt2prediction = unchunk_predictions(utt2prediction, config)

    logger.info("Collecting targets from dataset iterator.")
    utt2target = dict(collect_targets(labels, split2meta[data_conf["split"]]))
    return utt2prediction, utt2target


def evaluate_test_set(utt2prediction, utt2target, labels, config, test_conf):
    utt2prediction = generate_worst_case_predictions_for_missed_utterances(utt2prediction, utt2target, labels)
    write_predictions(utt2prediction, labels, config, test_conf["split"])
    return list(evaluate_metrics_for_predictions(
        utt2prediction,
        utt2target,
        test_conf["evaluate_metrics"],
        labels))


def evaluate_metrics_for_predictions(utt2prediction, utt2target, eval_confs, labels):
    import sklearn.metrics

    logger.info("Stacking predictions to numpy arrays")
    # Ensure true labels are always in the same order as in predictions
    predictions = np.stack([p for _, p in utt2prediction])
    min_score = np.amin(predictions)
    max_score = np.amax(predictions)
    true_labels_sparse = np.array([utt2target[u] for u, _ in utt2prediction])
    pred_labels_sparse = np.argmax(predictions, axis=1)

    def onehot(i):
        o = np.zeros(len(labels))
        o[i] = 1
        return o

    true_labels_dense = np.stack([onehot(t) for t in true_labels_sparse])
    logger.info(
            "Evaluating metrics on true labels of shape %s and predicted labels of shape %s."
            " Min prediction score %.3f max prediction score %.3f",
            true_labels_sparse.shape, pred_labels_sparse.shape, float(min_score), float(max_score))

    for metric in eval_confs:
        result = None
        if metric["name"].endswith("average_detection_cost"):
            logger.info("Evaluating minimum average detection cost")
            thresholds = np.linspace(min_score, max_score, metric.get("num_thresholds", 50))
            if metric["name"].startswith("sparse_"):
                cavg = lidbox.metrics.SparseAverageDetectionCost(len(labels), thresholds)
                cavg.update_state(true_labels_sparse, predictions)
            else:
                cavg = lidbox.metrics.AverageDetectionCost(len(labels), thresholds)
                cavg.update_state(true_labels_dense, predictions)
            result = float(cavg.result().numpy())
            logger.info("%s: %.6f", metric["name"], result)
        elif metric["name"].endswith("average_equal_error_rate"):
            #TODO sparse EER, generate one-hot true_labels
            logger.info("Evaluating average equal error rate")
            eer = np.zeros(len(labels))
            for l, label in enumerate(labels):
                # https://stackoverflow.com/a/46026962
                fpr, tpr, _ = sklearn.metrics.roc_curve(true_labels_dense[:,l], predictions[:,l])
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
        yield {"name": metric["name"], "result": result}


def _metrics_file_from_config(config, dataset_name):
    from lidbox.models.keras_utils import experiment_cache_from_config
    return os.path.join(
            experiment_cache_from_config(config),
            "predictions",
            dataset_name,
            "metrics.json")


def write_predictions(utt2prediction, labels, config, dataset_name):
    scores_file = os.path.join(
            lidbox.models.keras_utils.experiment_cache_from_config(config),
            "predictions",
            dataset_name,
            "scores")
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    logger.info("Writing %d predicted scores to '%s'", len(utt2prediction), scores_file)
    if os.path.exists(scores_file):
        logger.warning("Overwriting existing '%s'", scores_file)
    with open(scores_file, "w") as scores_f:
        print_predictions(utt2prediction, labels, file=scores_f)


def write_metrics(metrics, config, dataset_name):
    metrics_file = _metrics_file_from_config(config, dataset_name)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    logger.info("Writing evaluated metrics to '%s'", metrics_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)


def load_metrics(config, dataset_name):
    with open(_metrics_file_from_config(config, dataset_name)) as f:
        return json.load(f)


def allow_tf_gpu_memory_growth():
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
