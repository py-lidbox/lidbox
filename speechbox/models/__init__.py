import datetime
import functools
import importlib
import io
import os
import sys

import numpy as np
import tensorflow as tf

import speechbox.metrics
from speechbox.tf_data import without_metadata

# Check if the KerasWrapper instance has a tf.device string argument and use that when running the method, else let tf decide
def with_device(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.device_str:
            with tf.device(self.device_str):
                return method(self, *args, **kwargs)
        else:
            return method(self, *args, **kwargs)
    return wrapped

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

def parse_metrics(metrics, target_names):
    keras_metrics = []
    for m in metrics:
        metric = None
        kwargs = dict(m.get("kwargs", {}))
        if "cls" in m:
            metric = getattr(tf.keras.metrics, m["cls"])(**kwargs)
        else:
            name = m["name"]
            if name in ("avg_equal_error_rate", "avg_eer"):
                metric = speechbox.metrics.AverageEqualErrorRate(target_names, **kwargs)
            elif name == "avg_recall":
                metric = speechbox.metrics.AverageRecall(target_names, **kwargs)
            elif name == "avg_precision":
                metric = speechbox.metrics.AveragePrecision(target_names, **kwargs)
            elif name == "C_avg":
                metric = speechbox.metrics.AverageDetectionCost(target_names, **kwargs)
        assert metric is not None, "unknown metric: '{}'".format(m)
        keras_metrics.append(metric)
    return keras_metrics

class EpochModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval, filepath, **kwargs):
        self.epoch_interval = epoch_interval
        self.checkpoint_path = filepath

    def on_epoch_end(self, epoch, *args):
        if epoch and epoch % self.epoch_interval == 0:
            self.model.save(self.checkpoint_path.format(epoch=epoch, **logs))


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

    @classmethod
    def get_model_filepath(cls, basedir, model_id):
        return os.path.join(basedir, cls.__name__.lower() + '-' + model_id)

    def __init__(self, model_id, model_definition, device_str=None, tensorboard=None, early_stopping=None, checkpoints=None, other_callbacks=()):
        self.model_id = model_id
        self.device_str = device_str
        self.model = None
        self.initial_epoch = 0
        model_module = importlib.import_module("speechbox.models." + model_definition["name"])
        self.model_loader = functools.partial(model_module.loader, **model_definition.get("kwargs", {}))
        self.predict_fn = model_module.predict
        self.callbacks = []
        if tensorboard:
            self.tensorboard = tf.keras.callbacks.TensorBoard(**tensorboard)
            self.callbacks.append(self.tensorboard)
        if early_stopping:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping))
        if checkpoints:
            if "epoch_interval" in checkpoints:
                # This is for saving checkpoints at regular epoch intervals, regardless of the values that ModelCheckpoint is monitoring
                self.callbacks.append(EpochModelCheckpoint(checkpoints.pop("epoch_interval"), checkpoints["filepath"]))
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoints))
        for cb in other_callbacks:
            self.callbacks.append(getattr(tf.keras.callbacks, cb["cls"])(**cb.get("kwargs", {})))

    @with_device
    def to_disk(self, basedir):
        model_path = self.get_model_filepath(basedir, self.model_id)
        self.model.save(model_path, overwrite=True)
        return model_path

    @with_device
    def prepare(self, target_names, training_config):
        input_shape = training_config["input_shape"]
        self.model = self.model_loader(input_shape, len(target_names))
        opt_conf = training_config["optimizer"]
        opt_kwargs = opt_conf.get("kwargs", {})
        if "lr_scheduler" in opt_kwargs:
            lr_scheduler = opt_kwargs.pop("lr_scheduler")
            opt_kwargs["learning_rate"] = getattr(tf.keras.optimizers.schedules, lr_scheduler["cls"])(**lr_scheduler["kwargs"])
        optimizer = getattr(tf.keras.optimizers, opt_conf["cls"])(**opt_kwargs)
        loss_conf = training_config["loss"]
        loss = getattr(tf.keras.losses, loss_conf["cls"])(**loss_conf.get("kwargs", {}))
        if "metrics" in training_config:
            metrics = parse_metrics(training_config["metrics"], target_names)
        else:
            metrics = None
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )

    @with_device
    def load_weights(self, path):
        self.initial_epoch = int(parse_checkpoint_value(path, key="epoch"))
        self.model.load_weights(path)

    @with_device
    def fit(self, training_set, validation_set, model_config):
        return self.model.fit(
            without_metadata(training_set),
            callbacks=self.callbacks,
            class_weight=model_config.get("class_weight"),
            epochs=model_config["epochs"],
            initial_epoch=self.initial_epoch,
            shuffle=False,
            steps_per_epoch=model_config.get("steps_per_epoch"),
            validation_data=without_metadata(validation_set),
            validation_freq=model_config.get("validation_freq", 1),
            validation_steps=model_config.get("validation_steps"),
            verbose=model_config.get("verbose", 2),
        )

    @with_device
    def predict(self, testset):
        return self.predict_fn(self.model, testset)

    @with_device
    def count_params(self):
        return sum(layer.count_params() for layer in self.model.layers)

    def __str__(self):
        string_stream = io.StringIO()
        def print_to_stream(*args, **kwargs):
            kwargs["file"] = string_stream
            print(*args, **kwargs)
        self.model.summary(print_fn=print_to_stream)
        return string_stream.getvalue()
