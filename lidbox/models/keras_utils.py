"""
Wrappers on top of tf.keras to automate interactions using the lidbox config file.
"""
import datetime
import importlib
import io
import os
import sys
import time

import tensorflow as tf

import lidbox.metrics


MODELS_IMPORT_PATH = "lidbox.models."


def experiment_cache_from_config(config):
    experiment_config = config["sklearn_experiment"] if "sklearn_experiment" in config else config["experiment"]
    return os.path.join(
            experiment_config["cache_directory"],
            experiment_config["model"]["key"],
            experiment_config["name"])


def best_model_checkpoint_from_config(config):
    checkpoint_callbacks = [d for d in config["experiment"].get("callbacks", []) if d["cls"] == "ModelCheckpoint"]
    if checkpoint_callbacks:
        checkpoint_kwargs = checkpoint_callbacks[0].get("kwargs", {})
    if "filepath" in checkpoint_kwargs:
        checkpoints_dir = os.path.dirname(checkpoint_kwargs["filepath"])
    else:
        checkpoints_dir = os.path.join(experiment_cache_from_config(config), "checkpoints")
    return KerasWrapper.get_best_checkpoint_path(
            checkpoints_dir,
            key=checkpoint_kwargs.get("monitor"),
            mode=checkpoint_kwargs.get("mode"))


def parse_checkpoint_value(tf_checkpoint_path, key):
    return tf_checkpoint_path.split(key)[-1].split("__")[0].split(".hdf5")[0]


def init_metric_from_config(config):
    if config["cls"].endswith("AverageDetectionCost"):
        args = [config["threshold_linspace"][k] for k in ("start", "stop", "num")]
        thresholds = tf.linspace(*args).numpy()
        metric = getattr(lidbox.metrics, config["cls"])(config["N"], thresholds)
    else:
        metric = getattr(tf.keras.metrics, config["cls"])(**config.get("kwargs", {}))
    return metric


def init_callback_from_config(config, cache_dir):
    user_kwargs = config.get("kwargs", {})
    if config["cls"] == "ModelCheckpoint":
        default_checkpoint_format = "epoch{epoch:06d}__val_loss{val_loss:.12f}.hdf5"
        callback_kwargs = {
            "filepath": os.path.join(
                os.path.join(cache_dir, "checkpoints"),
                config.get("format", default_checkpoint_format))}
        callback_kwargs.update(user_kwargs)
        os.makedirs(os.path.dirname(callback_kwargs["filepath"]), exist_ok=True)
    elif config["cls"] == "TensorBoard":
        callback_kwargs = {
            "histogram_freq": 1,
            "log_dir": os.path.join(cache_dir, "tensorboard", str(int(time.time()))),
            "profile_batch": 0}
        callback_kwargs.update(user_kwargs)
        os.makedirs(callback_kwargs["log_dir"], exist_ok=True)
    else:
        callback_kwargs = user_kwargs
    if config["cls"] == "LearningRateDateLogger":
        callback = LearningRateDateLogger(**callback_kwargs)
    else:
        callback = getattr(tf.keras.callbacks, config["cls"])(**callback_kwargs)
    return callback


class LearningRateDateLogger(tf.keras.callbacks.Callback):
    def __init__(self, output_stream=sys.stdout, **kwargs):
        self.output_stream = output_stream

    def on_epoch_begin(self, epoch, *args):
        tf.print(
            str(datetime.datetime.now()),
            "-",
            self.model.optimizer.__class__.__name__,
            "learning rate:",
            self.model.optimizer._decayed_lr(tf.float32),
            summarize=-1,
            output_stream=self.output_stream)


class KerasWrapper:
    """
    Wrapper class over tf.keras models for automatic initialization from lidbox config files.
    """

    @staticmethod
    def get_best_checkpoint_path(checkpoints_dir, key=None, mode=None):
        if key is None:
            key = "epoch"
        checkpoints = [p.path for p in os.scandir(checkpoints_dir) if p.is_file() and p.name.endswith(".hdf5")]
        key_fn = lambda p: parse_checkpoint_value(p, key)
        best_path = None
        if checkpoints:
            if key == "epoch":
                # Greatest epoch value
                best_path = max(checkpoints, key=lambda p: int(key_fn(p)))
            else:
                assert mode in ("min", "max"), mode
                if mode == "min":
                    best_path = min(checkpoints, key=lambda p: float(key_fn(p)))
                else:
                    best_path = max(checkpoints, key=lambda p: float(key_fn(p)))
        return best_path

    @classmethod
    def get_model_filepath(cls, basedir, model_key):
        return os.path.join(basedir, cls.__name__.lower() + '-' + model_key)

    @classmethod
    def from_config(cls, config):
        experiment_cache = experiment_cache_from_config(config)
        os.makedirs(experiment_cache, exist_ok=True)
        now_str = str(int(time.time()))
        model_key = config["experiment"]["model"]["key"]
        model_module = importlib.import_module(MODELS_IMPORT_PATH + model_key)
        input_shape = config["experiment"]["input_shape"]
        output_shape = tf.squeeze(config["experiment"]["output_shape"])
        loader_kwargs = config["experiment"]["model"].get("kwargs", {})
        keras_model = model_module.loader(input_shape, output_shape, **loader_kwargs)
        opt_conf = config["experiment"]["optimizer"]
        opt_kwargs = opt_conf.get("kwargs", {})
        if "lr_scheduler" in opt_kwargs:
            lr_scheduler = opt_kwargs.pop("lr_scheduler")
            opt_kwargs["learning_rate"] = getattr(tf.keras.optimizers.schedules, lr_scheduler["cls"])(**lr_scheduler["kwargs"])
        optimizer = getattr(tf.keras.optimizers, opt_conf["cls"])(**opt_kwargs)
        loss_conf = config["experiment"]["loss"]
        loss = getattr(tf.keras.losses, loss_conf["cls"])(**loss_conf.get("kwargs", {}))
        metrics = [init_metric_from_config(c) for c in config["experiment"].get("metrics", [])]
        keras_model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics)
        callbacks = [init_callback_from_config(c, experiment_cache) for c in config["experiment"].get("callbacks", [])]
        return cls(keras_model, model_key, callbacks)

    @classmethod
    def from_config_as_embedding_extractor_fn(cls, config):
        experiment_cache = experiment_cache_from_config({"experiment":
            {"cache_directory": config["cache_directory"],
             "model": config["model"],
             "name": config["experiment_name"]}})
        model_key = config["model"]["key"]
        model_module = importlib.import_module(MODELS_IMPORT_PATH + model_key)
        input_shape = config["input_shape"]
        output_shape = tf.squeeze(config["output_shape"])
        loader_kwargs = config["model"].get("kwargs", {})
        keras_model = model_module.loader(input_shape, output_shape, **loader_kwargs)
        keras_model.load_weights(cls.get_best_checkpoint_path(
            os.path.join(experiment_cache, "checkpoints"),
            key=config["best_checkpoint"]["monitor"],
            mode=config["best_checkpoint"]["mode"]))
        as_extractor = getattr(model_module, "as_embedding_extractor")
        keras_model = as_extractor(keras_model)
        keras_model.trainable = False
        model_input = keras_model.inputs[0]
        extractor_fn = tf.function(
                lambda x: keras_model(x, training=False),
                input_signature=[tf.TensorSpec(model_input.shape, model_input.dtype)])
        return extractor_fn.get_concrete_function()

    def __init__(self, keras_model, model_key, callbacks):
        self.model_key = model_key
        self.keras_model = keras_model
        self.initial_epoch = 0
        self.callbacks = callbacks

    def to_disk(self, basedir):
        model_path = self.get_model_filepath(basedir, self.model_key)
        self.keras_model.save(model_path, overwrite=True)
        return model_path

    def load_weights(self, path):
        self.initial_epoch = int(parse_checkpoint_value(path, key="epoch"))
        self.keras_model.load_weights(path)

    def fit(self, training_dataset, validation_dataset, user_kwargs):
        kwargs = {
            "shuffle": False,
            "validation_freq": 1,
            "verbose": 2,
        }
        kwargs.update(user_kwargs)
        return self.keras_model.fit(
            training_dataset,
            validation_data=validation_dataset,
            callbacks=self.callbacks,
            initial_epoch=self.initial_epoch,
            **kwargs)

    def count_params(self):
        return sum(layer.count_params() for layer in self.keras_model.layers)

    def __str__(self):
        with io.StringIO() as sstream:
            def print_to_stream(*args, **kwargs):
                kwargs["file"] = sstream
                print(*args, **kwargs)
            self.keras_model.summary(print_fn=print_to_stream)
            return sstream.getvalue()
