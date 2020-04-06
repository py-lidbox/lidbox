# This script will be imported dynamically by lidbox.
# Function 'create_dataset' is used to create tf.data.Dataset instance for each split
# If defined, function 'train' is given the tf.data.Dataset instances for splits 'train' and 'validation'
import os
import lidbox.api


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
    Step = lidbox.api.Step
    cache_dir = os.path.join(config["cache"]["directory"], "features", split)
    tensorboard_summary_dir = os.path.join(config["cache"]["directory"], "dataset_tensorboard")
    cache_config = {
            "directory": cache_dir,
            "cache_key": "logmelspectrogram",
            "batch_size": config["cache"]["batch_size"]}
    return [
        # Create a tf.data.Dataset that contains all metadata, e.g. paths from utt2path and labels from utt2label etc.
        Step("initialize", {"labels": labels, "init_data": init_data}),
        # Load signals from all paths
        Step("load_audio", {}),
        # Drop empty signals
        Step("drop_empty", {}),
        # Compute WebRTC VAD decisions
        Step("compute_webrtc_vad", config["pre_process"]["webrtcvad"]),
        # Drop non-speech frames using computed decisions
        Step("apply_vad", {}),
        Step("drop_empty", {}),
        # Divide all signals into chunks
        Step("create_signal_chunks", config["pre_process"]["chunks"]),
        Step("drop_empty", {}),
        # Drop some unneeded keys from each element
        Step("drop_keys_in_set", {"keys": {"vad_is_speech", "duration", "path", "dataset"}}),
        # Apply feature extraction, uses GPU by default, change with 'device' key
        Step("extract_features", {"config": config["features"]}),
        # Drop unwanted samples
        Step("apply_filters", {"config": config["filters"]}),
        # Serialize all elements to disk
        Step("cache", cache_config),
        # Evaluate whole pipeline up to this point, this fills the cache
        Step("consume", {"log_interval": 10000}),
        # Add some samples to TensorBoard for inspection
        Step("consume_to_tensorboard", {"summary_dir": tensorboard_summary_dir, "config": config["show_samples"]}),
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
