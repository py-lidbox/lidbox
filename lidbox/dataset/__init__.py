"""
Modular building blocks for constructing a tf.data.Dataset pipeline from metadata files on the local machine.
"""
import logging
logger = logging.getLogger("dataset")

from .steps import VALID_STEP_FUNCTIONS


def from_steps(steps):
    ds = None
    if steps[0].key != "initialize":
        logger.critical("When constructing a dataset, the first step must be 'initialize' but it was '%s'. The 'initialize' step is needed for first loading all metadata such as the utterance_id to wavpath mappings.", steps[0].key)
        return
    for step_num, step in enumerate(steps, start=1):
        step_fn = VALID_STEP_FUNCTIONS.get(step.key)
        if step_fn is None:
            logger.critical("Unknown step '%s' when constructing dataset", step.key)
            return
        logger.info("Applying step number %d: '%s'", step_num, step.key)
        ds = step_fn(ds, **step.kwargs)
    return ds
