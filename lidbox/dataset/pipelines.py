"""
Default data pipelines constructed from dataset steps.
This module can be replaced by a custom script using key 'user_script' in the config file.
"""
import os
from lidbox.dataset.steps import Step
from lidbox.models.keras_utils import experiment_cache_from_config


def _get_cache_steps(config, split):
    cache_config = {
            "directory": os.path.join(config["directory"], "dataset", split),
            "cache_key": config.get("key"),
            "batch_size": config["batch_size"]}
    yield Step("cache", cache_config)
    if config.get("consume", True):
        yield Step("consume", {"log_interval": config.get("log_interval", -1)})


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
    # Configure steps to create dataset iterator
    steps = [
        # Create a tf.data.Dataset that contains all metadata, e.g. paths from utt2path and labels from utt2label etc.
        Step("initialize", {"labels": labels, "init_data": init_data}),
    ]
    if "post_initialize" in config:
        # "Pre-pre-process" all metadata before any signals are read
        if "file_limit" in config["post_initialize"]:
            steps.append(Step("lambda", {"fn": lambda ds: ds.take(config["post_initialize"]["file_limit"])}))
        if "shuffle_buffer_size" in config["post_initialize"]:
            # Shuffle all files
            steps.append(Step("shuffle", {"buffer_size": config["post_initialize"]["shuffle_buffer_size"]}))
        if "binary_classification" in config["post_initialize"]:
            # Convert all labels to binary classification
            steps.append(Step("convert_to_binary_classification", {"positive_class": config["post_initialize"]["binary_classification"]}))
        if config["post_initialize"].get("check_wav_headers", False):
            steps.append(Step("drop_invalid_wavs", {}))
    if "features" in config and config["features"]["type"] == "kaldi":
        # Features will be imported from Kaldi files, assume no signals should be loaded
        pass
    else:
        # Assume all features will be extracted from signals
        steps.extend([
            # Load signals from all paths
            Step("load_audio", {"num_prefetch": config.get("post_initialize", {"num_prefetched_signals": None})["num_prefetched_signals"]}),
            # Drop empty signals
            Step("drop_empty", {})])
    if "pre_process" in config:
        # Pre-processing before feature extraction has been defined in the config file
        if "filters" in config["pre_process"]:
            # Drop unwanted signals
            steps.append(Step("apply_filters", {"config": config["pre_process"]["filters"]}))
        if "webrtcvad" in config["pre_process"] or "rms_vad" in config["pre_process"]:
            # Voice activity detection
            if "webrtcvad" in config["pre_process"]:
                # Compute WebRTC VAD decisions
                steps.append(Step("compute_webrtc_vad", config["pre_process"]["webrtcvad"]))
            elif "rms_vad" in config["pre_process"]:
                # Compute VAD decisions by comparing the RMS value of each VAD frame to the mean RMS value over each signal
                steps.append(Step("compute_rms_vad", config["pre_process"]["rms_vad"]))
            steps.extend([
                # Drop non-speech frames using computed decisions
                Step("apply_vad", {}),
                # Some signals might contain only non-speech frames
                Step("drop_empty", {}),
            ])
        if "repeat_too_short_signals" in config["pre_process"]:
            # Repeat all signals until they are of given length
            steps.append(Step("repeat_too_short_signals", config["pre_process"]["repeat_too_short_signals"]))
        if "augment" in config["pre_process"]:
            augment_configs = [conf for conf in config["pre_process"]["augment"] if conf["split"] == split]
            # Apply augmentation only if this dataset split was specified to be augmented
            if augment_configs:
                steps.append(Step("augment_signals", {"augment_configs": augment_configs}))
        if "chunks" in config["pre_process"]:
            # Dividing signals into fixed length chunks
            steps.append(Step("create_signal_chunks", config["pre_process"]["chunks"]))
        # TODO not yet supported
        # if "random_chunks" in config["pre_process"]:
        if "cache" in config["pre_process"]:
            steps.extend(_get_cache_steps(config["pre_process"]["cache"], split))
    if "features" in config:
        # Load features
        if config["features"]["type"] == "kaldi":
            # Pre-extracted Kaldi features will be used as input
            steps.append(
                # Use the 'kaldi_ark_key' to load contents from an external Kaldi archive file and drop Kaldi metadata
                Step("load_kaldi_data", {"shape": config["features"]["kaldi"]["shape"]}))
        else:
            # Features will be extracted from 'signal' and stored under 'input'
            # Uses GPU by default, can be changed with the 'device' key
            steps.append(Step("extract_features", {"config": config["features"]}))
    if "post_process" in config:
        if "filters" in config["post_process"]:
            # Drop unwanted features
            steps.append(Step("apply_filters", {"config": config["post_process"]["filters"]}))
        if "chunks" in config["post_process"]:
            # Dividing inputs into fixed length chunks
            steps.append(Step("create_input_chunks", config["post_process"]["chunks"]))
        if "normalize" in config["post_process"]:
            steps.append(Step("normalize", {"config": config["post_process"]["normalize"]}))
        if "shuffle_buffer_size" in config["post_process"]:
            steps.append(Step("shuffle", {"buffer_size": config["post_process"]["shuffle_buffer_size"]}))
        if "tensorboard" in config["post_process"]:
            tensorboard_config = {
                    "summary_dir": os.path.join(
                        experiment_cache_from_config(config),
                        "tensorboard",
                        "dataset",
                        split),
                    "config": config["post_process"]["tensorboard"]}
            # Add some samples to TensorBoard for inspection
            steps.append(Step("consume_to_tensorboard", tensorboard_config))
        if "remap_keys" in config["post_process"]:
            steps.append(Step("remap_keys", {"new_keys": config["post_process"]["remap_keys"]}))
        if "cache" in config["post_process"]:
            steps.extend(_get_cache_steps(config["post_process"]["cache"], split))
    # TODO convert to binary classification here
    # TODO pre_training config key
    if "experiment" in config:
        # Check this split should be shuffled before training
        for experiment_conf in config["experiment"]["data"].values():
            if experiment_conf["split"] == split and "shuffle_buffer_size" in experiment_conf:
                steps.append(Step("shuffle", {"buffer_size": experiment_conf["shuffle_buffer_size"]}))
                break
    if "embeddings" in config:
        steps.append(Step("extract_embeddings", {"config": config["embeddings"]}))
        if "remap_keys" in config["embeddings"]:
            steps.append(Step("remap_keys", {"new_keys": config["embeddings"]["remap_keys"]}))
        if "cache" in config["embeddings"]:
            steps.extend(_get_cache_steps(config["embeddings"]["cache"], split))
    return steps
