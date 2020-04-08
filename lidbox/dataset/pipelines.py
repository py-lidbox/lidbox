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
    # Configure steps to create dataset iterator
    Step = lidbox.api.Step
    steps = []
    steps.extend([
        # Create a tf.data.Dataset that contains all metadata, e.g. paths from utt2path and labels from utt2label etc.
        Step("initialize", {"labels": labels, "init_data": init_data}),
        # Load signals from all paths
        Step("load_audio", {}),
        # Drop empty signals
        Step("drop_empty", {}),
    ])
    if "pre_process" in config:
        # Pre-processing before feature extraction has been defined in the config file
        if "filters" in config["pre_process"]:
            # Drop unwanted signals
            steps.extend([
                Step("apply_filters", {"config": config["pre_process"]["filters"]}),
            ])
        if "webrtcvad" in config["pre_process"]:
            # Voice activity detection
            steps.extend([
                # Compute WebRTC VAD decisions
                Step("compute_webrtc_vad", config["pre_process"]["webrtcvad"]),
                Step("reduce_stats", {"statistic": "vad_ratio"}),
                # Drop non-speech frames using computed decisions
                Step("apply_vad", {}),
                # Some signals might contain only non-speech frames
                Step("drop_empty", {}),
            ])
        if "chunks" in config["pre_process"]:
            # Dividing signals into fixed length chunks
            steps.extend([
                Step("create_signal_chunks", config["pre_process"]["chunks"]),
            ])
        # TODO not yet supported
        # if "random_chunks" in config["pre_process"]:
    if "features" in config:
        # Feature extraction
        steps.extend([
            # Apply feature extraction, uses GPU by default, change with 'device' key
            Step("extract_features", {"config": config["features"]}),
        ])
    if "post_process" in config:
        if "filters" in config["post_process"]:
            # Drop unwanted features
            steps.extend([
                Step("apply_filters", {"config": config["post_process"]["filters"]}),
            ])
        if "chunks" in config["post_process"]:
            # Dividing inputs into fixed length chunks
            steps.extend([
                Step("create_input_chunks", config["post_process"]["chunks"]),
            ])
    if "cache" in config:
        cache_root = config["cache"]["directory"]
        cache_config = {
                "directory": os.path.join(cache_root, "features", split),
                "cache_key": config["cache"].get("key"),
                "batch_size": config["cache"]["batch_size"]}
        # Serialize all elements to disk and eagerly evaluate whole pipeline
        steps.extend([
            Step("cache", cache_config),
            Step("consume", {"log_interval": 10000}),
        ])
        if "show_samples" in config:
            tensorboard_summary_dir = os.path.join(cache_root, "dataset_tensorboard", split)
            # Add some samples to TensorBoard for inspection
            steps.extend([
                Step("consume_to_tensorboard", {"summary_dir": tensorboard_summary_dir, "config": config["show_samples"]}),
            ])
    return steps
