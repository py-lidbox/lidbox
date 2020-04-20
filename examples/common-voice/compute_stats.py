"""
If a script contains a function named 'modify_steps', it can be used to mutate the dataset steps pipeline.
Here we insert some steps that reduce the dataset into a single statistic.
"""
from lidbox.dataset.steps import Step


batch_size = 1000

def modify_steps(steps, split, labels, init_data, config):
    # Find the step where VAD decisions are added
    i = [i for i, s in enumerate(steps) if s.key == "compute_webrtc_vad"][-1]
    # Compute frequency of dropped and kept frames by the VAD decisions
    steps.insert(i + 1, Step("reduce_stats", {"statistic": "vad_ratio"}))
    # Find the step where features are extracted
    i = [i for i, s in enumerate(steps) if s.key == "extract_features"][-1]
    # Compute frequency of NaN and inf values in the features
    steps.insert(i + 1, Step("reduce_stats", {"statistic": "num_non_finite", "key": "input", "axis": [1, 2], "batch_size": batch_size}))
    # Compute min, max and mean of all scalar values of the features
    steps.insert(i + 2, Step("reduce_stats", {"statistic": "min_max_mean", "key": "input", "batch_size": batch_size}))
    # Compute number of different shapes
    steps.insert(i + 3, Step("reduce_stats", {"statistic": "size_counts", "key": "input", "ndims": 3, "batch_size": batch_size}))
    # Drop remaining steps, we only need to compute the stats
    return steps[:i + 4]
