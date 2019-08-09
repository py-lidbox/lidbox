from datetime import datetime
import os

import numpy as np
import tensorflow as tf


class KerasWrapper:

    @classmethod
    def get_model_filepath(cls, basedir, model_id):
        return os.path.join(basedir, cls.__name__.lower() + '-' + model_id)

    #TODO checkpoints, weights, everything from disk so that epochs are resumable
    @classmethod
    def from_disk(cls, basedir, model_id, *init_args, **init_kwargs):
        m = cls(model_id, *init_args, **init_kwargs)
        m.model = tf.keras.models.load_model(cls.get_model_filepath(basedir, model_id))
        return m

    def __init__(self, model_id, log_dir=None, early_stopping=None):
        self.model_id = model_id
        self.model = None
        self.callbacks = []
        if log_dir:
            utcnow_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            log_dir_today = os.path.join(log_dir, utcnow_str)
            self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir_today))
        #TODO
        if early_stopping:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping))

    def to_disk(self, basedir):
        model_path = self.get_model_filepath(basedir, self.model_id)
        self.model.save(model_path, overwrite=True)
        return model_path

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

    def fit(self, training_set, validation_set, model_config):
        self.model.fit(
            training_set,
            validation_data=validation_set,
            epochs=model_config["epochs"],
            steps_per_epoch=model_config["steps_per_epoch"],
            validation_steps=model_config["validation_steps"],
            verbose=model_config.get("verbose", 2),
            callbacks=self.callbacks,
        )

    def evaluate(self, test_set, model_config):
        return self.model.evaluate(
            test_set,
            steps=model_config["validation_steps"],
            verbose=model_config.get("verbose", 2)
        )

    def predict(self, utterances):
        return self.model.predict(utterances)

    def evaluate_confusion_matrix(self, utterances, real_labels):
        predicted_labels = np.int8(self.predict(utterances).argmax(axis=1))
        real_labels = np.int8(real_labels.argmax(axis=1))
        cm = tf.confusion_matrix(real_labels, predicted_labels)
        with tf.Session() as session:
            return cm.eval(session=session)
