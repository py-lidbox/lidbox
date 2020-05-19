"""
Modular building blocks for constructing a tf.data.Dataset pipeline from metadata files on the local machine.

See lidbox.dataset.steps for the step definitions.
"""
import logging
logger = logging.getLogger(__name__)

from lidbox.dataset.steps import from_steps

def _allow_tf_gpu_memory_growth():
    import tensorflow as tf
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

try:
    _allow_tf_gpu_memory_growth()
except RuntimeError:
    logger.exception("Failed to allow memory growth for GPU devices, TensorFlow might allocate all available GPU memory. The exception was:")
