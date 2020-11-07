import itertools
import numpy as np
import pandas as pd


def predict_with_model(model, test_ds, batch_size=1):
    ids = []
    predictions = []

    for x in test_ds.batch(batch_size).as_numpy_iterator():
        ids.extend(id.decode("utf-8") for id in x["id"])
        predictions.extend(p.numpy() for p in model(x["input"], training=False))

    return (pd.DataFrame.from_dict({
                "id": ids,
                "prediction": predictions})
            .set_index("id", drop=True, verify_integrity=True)
            .sort_index())


def chunk_parent_id(chunk_id):
    return chunk_id.rsplit('-', 1)[0]


def merge_chunk_predictions(chunk_predictions, merge_fn=np.mean):
    return (chunk_predictions
            .groupby(chunk_parent_id)
            .agg(lambda row: merge_fn(np.array(row), axis=0)))


# TODO
# 1. load metadata
# 2. merge, preprocess, update, prepare, meta
# 3. create tf.data.Dataset
# 4. build pipeline
# 5. train model
# 6. serialize model
# 7. compute metrics
