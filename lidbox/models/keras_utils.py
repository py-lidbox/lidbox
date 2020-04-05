import functools
import importlib
import io
import os
import time

import tensorflow as tf

import lidbox.metrics


MODELS_IMPORT_PATH = "lidbox.models."


def parse_checkpoint_value(tf_checkpoint_path, key):
    return tf_checkpoint_path.split(key)[-1].split("__")[0].split(".hdf5")[0]


def get_best_checkpoint(checkpoints, key="epoch", mode=None):
    key_fn = lambda p: parse_checkpoint_value(p, key)
    if key == "epoch":
        # Greatest epoch value
        return max(checkpoints, key=lambda p: int(key_fn(p)))
    else:
        assert mode in ("min", "max"), mode
        if mode == "min":
            return min(checkpoints, key=lambda p: float(key_fn(p)))
        elif mode == "max":
            return max(checkpoints, key=lambda p: float(key_fn(p)))


def init_metric_from_config(config):
    if config["cls"].endswith("AverageDetectionCost"):
        N = output_shape[0]
        args = [config["threshold_linspace"][k] for k in ("start", "stop", "num")]
        thresholds = tf.linspace(*args).numpy()
        metric = getattr(lidbox.metrics, config["cls"])(N, thresholds)
    else:
        metric = getattr(tf.keras.metrics, config["cls"])(**config.get("kwargs", {})))
    return metric


def init_callback_from_config(config, cache_dir):
    user_kwargs = config.get("kwargs", {})
    if config["cls"] == "ModelCheckpoint":
        callback_kwargs = {
            "filepath": os.path.join(
                os.path.join(cache_dir, "checkpoints"),
                config.get("format", "epoch{epoch:06d}.hdf5"))}
        callback_kwargs.update(user_kwargs)
        os.makedirs(os.path.dirname(callback_kwargs["filepath"]), exist_ok=True)
    elif config["cls"] = "TensorBoard":
        callback_kwargs = {
            "histogram_freq": 1,
            "log_dir": os.path.join(cache_dir, "tensorboard", str(int(time.time()))),
            "profile_batch": 0}
        callback_kwargs.update(user_kwargs)
        os.makedirs(cb_kwargs["log_dir"], exist_ok=True)
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
            self.keras_model.optimizer.__class__.__name__,
            "learning rate:",
            self.keras_model.optimizer._decayed_lr(tf.float32),
            summarize=-1,
            output_stream=self.output_stream)


class KerasWrapper:
    """
    Wrapper class over tf.keras models for automatic initialization from lidbox config files.
    """

    @classmethod
    def get_model_filepath(cls, basedir, model_key):
        return os.path.join(basedir, cls.__name__.lower() + '-' + model_key)

    @classmethod
    def from_config(cls, config):
        model_key = config["experiment"]["model"]["key"]
        experiment_name = config["experiment"]["name"]
        experiment_cache = os.path.join(config["cache"]["directory"], model_key, experiment_name)
        os.makedirs(experiment_cache, exist_ok=True)
        now_str = str(int(time.time()))
        model_module = importlib.import_module(MODELS_IMPORT_PATH + model_key)
        input_shape = config["experiment"]["input_shape"]
        output_shape = config["experiment"]["output_shape"]
        loader_kwargs = config["experiment"]["model"].get("kwargs", {})
        keras_model = model_module.loader(input_shape, output_shape, **loader_kwargs)
        opt_conf = training_config["optimizer"]
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
        return cls(keras_model, model_key, callbacks, model_module.predict)

    def __init__(self, keras_model, model_key, callbacks, predict_fn):
        self.model_key = model_key
        self.keras_model = keras_model
        self.predict_fn = predict_fn
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

    def predict(self, dataset):
        return self.predict_fn(self.keras_model, dataset)

    def count_params(self):
        return sum(layer.count_params() for layer in self.keras_model.layers)

    def __str__(self):
        string_stream = io.StringIO()
        def print_to_stream(*args, **kwargs):
            kwargs["file"] = string_stream
            print(*args, **kwargs)
        self.keras_model.summary(print_fn=print_to_stream)
        return string_stream.getvalue()
