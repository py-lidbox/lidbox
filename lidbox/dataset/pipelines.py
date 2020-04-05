from lidbox.dataset.steps import Step

def extract_and_cache(config):
    return [
        Step("initialize", {
            "metadata_paths": config["paths"],
            "file_limit": -1,
            "shuffle_files": [],
            "labels": dataset_config[dataset_key]["labels"]}),
        Step("load_audio", {}),
        Step("drop_empty", {}),
        Step("apply_filters", {"config": config["filters"]}),
        Step("compute_vad", config["pre_process"]["webrtcvad"]),
        Step("apply_vad", {}),
        Step("drop_empty", {}),
        Step("create_signal_chunks", config["pre_process"]["chunks"]),
        Step("drop_empty", {}),
        Step("drop_keys_in_set", {"keys": {"vad_is_speech", "duration", "path", "dataset"}}),
        Step("extract_features", {"config": config["features"]}),
        Step("cache", {"cache_dir": cache_dir, "batch_size": config["cache"]["batch_size"]}),
        Step("consume", {"log_interval": 10000}),
        Step("consume_to_tensorboard", {"summary_dir": tensorboard_summary_dir, "config": config["show_samples"]}),
    ]
