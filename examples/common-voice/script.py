# This script will be imported dynamically by lidbox.
# Function 'create_dataset' is used to create tf.data.Dataset instance for each split
# If defined, function 'train' is given the tf.data.Dataset instances for splits 'train' and 'validation'
import os
import lidbox.api


# See lidbox.dataset.pipelines.create_dataset
def create_dataset(split, labels, init_data, config):
    """
    split:
        Split key.
    labels:
        All labels from all datasets.
    init_data:
        All metadata by split from all datasets.
    config:
        Contents of the lidbox config file, unmodified.
    """
    return [
        # Create a tf.data.Dataset that contains all metadata, e.g. paths from utt2path and labels from utt2label etc.
        Step("initialize", {"labels": labels, "init_data": init_data}),
        # Load signals from all paths
        Step("load_audio", {}),
        # etc.
        # etc.
    ]


# Training can be defined in a function 'train'
# def train(train_ds, validation_ds, config):
#     keras_wrapper = lidbox.api.KerasWrapper.from_config(config)
#     print("Model initialized:\n{}".format(str(keras_wrapper)))
#     return keras_wrapper.fit(train_ds, validation_ds, config["experiment"].get("keras_fit_kwargs", {}))


# Prediction with a trained model can be defined in a function 'predict'
# def predict(test_ds, config):
#     keras_wrapper = lidbox.api.KerasWrapper.from_config(config)
#     print("Model initialized:\n{}".format(str(keras_wrapper)))
#     best_checkpoint = lidbox.api.best_model_checkpoint_from_config(config)
#     print("Loading weights from checkpoint file '{}'".format(best_checkpoint))
#     keras_wrapper.load_weights(best_checkpoint)
#     return keras_wrapper.keras_model.predict(test_ds)
