import functools
import io
import os

import numpy as np
import tensorflow as tf


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


class KerasWrapper:

    @classmethod
    def get_model_filepath(cls, basedir, model_id):
        return os.path.join(basedir, cls.__name__.lower() + '-' + model_id)

    def __init__(self, model_id, device_str=None, tensorboard=None, early_stopping=None, checkpoints=None):
        self.model_id = model_id
        self.device_str = device_str
        self.model = None
        self.callbacks = []
        if tensorboard:
            self.callbacks.append(tf.keras.callbacks.TensorBoard(**tensorboard))
        if early_stopping:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping))
        if checkpoints:
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoints))

    @with_device
    def to_disk(self, basedir):
        model_path = self.get_model_filepath(basedir, self.model_id)
        self.model.save(model_path, overwrite=True)
        return model_path

    @with_device
    def prepare(self, features_meta, model_config):
        input_shape = features_meta["sequence_length"], features_meta["num_features"]
        num_cells = model_config["num_cells"]
        num_labels = features_meta["num_labels"]
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(num_cells, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.LSTM(num_cells),
            tf.keras.layers.Dense(num_labels, activation='softmax')
        ])
        self.model.compile(
            loss=model_config["loss"],
            optimizer=model_config["optimizer"],
            metrics=model_config["metrics"]
        )

    @with_device
    def load_weights(self, path):
        self.model.load_weights(path)

    @with_device
    def fit(self, training_set, validation_set, model_config):
        self.model.fit(
            training_set,
            validation_data=validation_set,
            epochs=model_config["epochs"],
            steps_per_epoch=model_config.get("steps_per_epoch"),
            validation_steps=model_config.get("validation_steps"),
            verbose=model_config.get("verbose", 2),
            callbacks=self.callbacks,
        )

    @with_device
    def evaluate(self, test_set, model_config):
        return self.model.evaluate(
            test_set,
            steps=model_config.get("validation_steps"),
            verbose=model_config.get("verbose", 2)
        )

    @with_device
    def predict(self, utterances):
        expected_num_labels = self.model.layers[-1].output_shape[-1]
        predictions = np.zeros((len(utterances), expected_num_labels))
        for i, sequences in enumerate(utterances):
            predictions[i] = self.model.predict(sequences).mean(axis=0)
        return predictions

    @with_device
    def evaluate_confusion_matrix(self, utterances, real_labels):
        predicted_labels = np.int8(self.predict(utterances).argmax(axis=1))
        real_labels = np.int8(real_labels)
        return tf.math.confusion_matrix(real_labels, predicted_labels).numpy()

    def __str__(self):
        string_stream = io.StringIO()
        def print_to_stream(*args, **kwargs):
            kwargs["file"] = string_stream
            print(*args, **kwargs)
        self.model.summary(print_fn=print_to_stream)
        return string_stream.getvalue()
