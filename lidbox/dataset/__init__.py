"""
Modular building blocks for constructing a tf.data.Dataset pipeline from metadata files on the local machine.

See lidbox.dataset.steps for the step definitions.
"""
import io
import logging
logger = logging.getLogger("dataset")

import yaml

import lidbox
import lidbox.dataset

def _allow_tf_gpu_memory_growth():
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

try:
    _allow_tf_gpu_memory_growth()
except RuntimeError:
    logger.exception("Failed to allow memory growth for GPU devices, TensorFlow might allocate all available GPU memory. The exception was:")


def from_steps(steps):
    ds = None
    if steps[0].key != "initialize":
        logger.critical("When constructing a dataset, the first step must be 'initialize' but it was '%s'. The 'initialize' step is needed for first loading all metadata such as the utterance_id to wavpath mappings.", steps[0].key)
        return
    for step_num, step in enumerate(steps, start=1):
        step_fn = lidbox.dataset.steps.VALID_STEP_FUNCTIONS.get(step.key)
        if step_fn is None:
            logger.error("Skipping unknown step '%s'.", step.key)
            continue
        logger.info("Applying step number %d: '%s'.", step_num, step.key)
        ds = step_fn(ds, **step.kwargs)
        if ds is None:
            logger.critical("Failed to apply step '%s', stopping.", step.key)
            return
    return ds
