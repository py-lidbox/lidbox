"""
High-level utilities and wrappers on top of high-level APIs of other libraries.
"""
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing
import tensorflow as tf

import lidbox.metrics


def predict_with_model(model, ds):
    """
    Map callable model over all batches in ds, predicting values for each element at key 'input'.
    """
    predict_fn = lambda x: (x["id"], model(x["input"], training=False))
    pred_ds = ds.map(predict_fn).unbatch()
    predictions = [(id.decode("utf-8"), pred) for id, pred in pred_ds.as_numpy_iterator()]

    return (pd.DataFrame.from_dict({
                "id": (id for id, _ in predictions),
                "prediction": (pred for _, pred in predictions)})
            .set_index("id", drop=True, verify_integrity=True)
            .sort_index())


def chunk_parent_id(chunk_id):
    return chunk_id.rsplit('-', 1)[0]


def merge_chunk_predictions(chunk_predictions, merge_fn=np.mean):
    return (chunk_predictions
            .groupby(chunk_parent_id)
            .agg(lambda row: merge_fn(np.array(row), axis=0)))


def classification_report(true_sparse, pred_dense, label2target, dense2sparse_fn=None, num_cavg_thresholds=100):
    """
    Compute classification metrics on a given vector of true labels (sparse) and predicted scores (dense/onehot).
    """
    if dense2sparse_fn is None:
        dense2sparse_fn = lambda pred: pred.argmax(axis=1)
    pred_sparse = dense2sparse_fn(pred_dense)

    report = sklearn.metrics.classification_report(
                    true_sparse,
                    pred_sparse,
                    labels=list(range(len(label2target))),
                    target_names=label2target,
                    output_dict=True,
                    zero_division=0)

    cavg_thresholds = np.linspace(
            pred_dense.min(),
            pred_dense.max(),
            num_cavg_thresholds)
    cavg = lidbox.metrics.SparseAverageDetectionCost(len(label2target), cavg_thresholds)
    cavg.update_state(true_sparse, pred_dense)
    report["avg_detection_cost"] = float(cavg.result().numpy())

    def to_dense(target):
        v = np.zeros(len(label2target))
        v[target] = 1
        return v

    true_dense = np.stack([to_dense(t) for t in true_sparse])

    eer = np.zeros(len(label2target))
    for l, label in enumerate(label2target):
        # https://stackoverflow.com/a/46026962
        fpr, tpr, _ = sklearn.metrics.roc_curve(true_dense[:,l], pred_dense[:,l])
        fnr = 1 - tpr
        eer[l] = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    report["avg_equal_error_rate"] = float(eer.mean())
    for label, i in label2target.items():
        report[label]["equal_error_rate"] = eer[i]

    report["confusion_matrix"] = sklearn.metrics.confusion_matrix(true_sparse, pred_sparse)

    # TODO convert to multi-level pandas.DataFrame by separating language metrics from summary metrics
    return report


def evaluate_testset_with_model(model, test_ds, test_meta, lang2target):
    """
    Utility for calling predict_with_model followed by classification_report.
    """
    with tf.device("GPU"):
        utt2pred = predict_with_model(model, test_ds)
    test_meta = test_meta.join(utt2pred, how="outer")
    assert not test_meta.isna().any(axis=None), "Failed to join predictions from test_ds with given test_meta dataframe: set of utterance ids is not equal"

    true_sparse = test_meta.target.to_numpy(np.int32)
    pred_dense = np.stack(test_meta.prediction)

    return classification_report(true_sparse, pred_dense, lang2target)


def model2function(model):
    model_input = model.inputs[0]
    model_fn = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec(model_input.shape, model_input.dtype)])
    return model_fn.get_concrete_function()
