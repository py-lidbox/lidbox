import functools
import importlib
import io
import os

import numpy as np
import tensorflow as tf

import speechbox.metrics

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

def get_best_checkpoint(checkpoints, key="epoch"):
    key_fn = lambda p: parse_checkpoint_value(p, key)
    if key == "epoch":
        # Greatest epoch value
        return max(checkpoints, key=lambda p: int(key_fn(p)))
    elif key == "val_loss":
        # Smallest validation loss value
        return min(checkpoints, key=lambda p: float(key_fn(p)))
    elif key == "val_accuracy":
        # Greatest validation accuracy value
        return max(checkpoints, key=lambda p: float(key_fn(p)))

def parse_metrics(metrics):
    keras_metrics = []
    for m in metrics:
        metric = None
        name = m["name"]
        kwargs = m.get("kwargs", {})
        if name == "accuracy":
            #FIXME why aren't Accuracy instances working?
            # metric = tf.keras.metrics.Accuracy()
            metric = name
        elif name == "precision":
            metric = tf.keras.metrics.Precision(**kwargs)
        elif name == "recall":
            metric = tf.keras.metrics.Recall(**kwargs)
        elif name == "equal_error_rate":
            metric = speechbox.metrics.OneHotAvgEER(kwargs.pop("num_classes"), **kwargs)
        elif name == "C_avg":
            metric = speechbox.metrics.AverageDetectionCost(**kwargs)
        assert metric is not None, "metric not implemented: '{}'".format(m)
        keras_metrics.append(metric)
    return keras_metrics

class EpochModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval, filepath, **kwargs):
        self.epoch_interval = epoch_interval
        self.checkpoint_path = filepath

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        if epoch and epoch % self.epoch_interval == 0:
            self.model.save(self.checkpoint_path.format(epoch=epoch, **logs))


class KerasWrapper:

    @classmethod
    def get_model_filepath(cls, basedir, model_id):
        return os.path.join(basedir, cls.__name__.lower() + '-' + model_id)

    def __init__(self, model_id, model_definition, device_str=None, tensorboard=None, early_stopping=None, checkpoints=None):
        self.model_id = model_id
        self.device_str = device_str
        self.model = None
        self.initial_epoch = 0
        import_path = "speechbox.models." + model_definition["name"]
        self.model_loader = functools.partial(importlib.import_module(import_path).loader, **model_definition["kwargs"])
        self.callbacks = []
        if tensorboard:
            self.callbacks.append(tf.keras.callbacks.TensorBoard(**tensorboard))
        if early_stopping:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping))
        if checkpoints:
            if "epoch_interval" in checkpoints:
                self.callbacks.append(EpochModelCheckpoint(checkpoints.pop("epoch_interval"), checkpoints["filepath"]))
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoints))

    @with_device
    def to_disk(self, basedir):
        model_path = self.get_model_filepath(basedir, self.model_id)
        self.model.save(model_path, overwrite=True)
        return model_path

    def enable_dataset_logger(self, dataset_name, dataset):
        tensorboard = [callback for callback in self.callbacks if "TensorBoard" in callback.__class__.__name__]
        assert len(tensorboard) == 1, "TensorBoard is not enabled for model or it has too many TensorBoard instances, there is nowhere to write the output of the logged metrics"
        tensorboard = tensorboard[0]
        metrics_dir = os.path.join(tensorboard.log_dir, "dataset", dataset_name)
        summary_writer = tf.summary.create_file_writer(metrics_dir)
        summary_writer.set_as_default()
        def inspect_batches(batch_idx, batch):
            inputs, targets = batch[:2]
            print("\nLogger enabled for dataset '{}', batch data will be written as histograms for TensorBoard".format(dataset_name))
            # Every value of all examples in this batch flattened to a single dimension
            tf.summary.histogram("{}-inputs".format(dataset_name), tf.reshape(inputs, [-1]), step=batch_idx)
            # Index of every one-hot encoded target in this batch
            tf.summary.histogram("{}-targets".format(dataset_name), tf.math.argmax(targets, 1), step=batch_idx)
            return batch
        dataset = dataset.enumerate().map(inspect_batches)
        return metrics_dir, dataset

    @with_device
    def prepare(self, features_meta, training_config):
        input_shape = features_meta["feat_vec_shape"]
        output_shape = features_meta["num_labels"]
        self.model = self.model_loader(input_shape, output_shape)
        opt_conf = training_config["optimizer"]
        optimizer = getattr(tf.keras.optimizers, opt_conf["cls"])(**opt_conf.get("kwargs", {}))
        self.model.compile(
            loss=training_config["loss"],
            optimizer=optimizer,
            metrics=parse_metrics(training_config["metrics"])
        )

    @with_device
    def load_weights(self, path):
        self.initial_epoch = int(parse_checkpoint_value(path, key="epoch"))
        self.model.load_weights(path)

    @with_device
    def fit(self, training_set, validation_set, model_config):
        self.model.fit(
            training_set,
            callbacks=self.callbacks,
            class_weight=model_config.get("class_weight"),
            epochs=model_config["epochs"],
            initial_epoch=self.initial_epoch,
            shuffle=False,
            steps_per_epoch=model_config.get("steps_per_epoch"),
            validation_data=validation_set,
            validation_steps=model_config.get("validation_steps"),
            verbose=model_config.get("verbose", 2),
        )

    @with_device
    def predict(self, utterances):
        #TODO compute all inside the tf graph
        expected_num_labels = self.model.layers[-1].output_shape[-1]
        predictions = np.zeros((len(utterances), expected_num_labels))
        for i, sequences in enumerate(utterances):
            prob_by_frame = self.model.predict(sequences)
            predictions[i] = prob_by_frame.mean(axis=0)
        return predictions

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
