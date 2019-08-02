"""File IO."""
import hashlib

import librosa
import tensorflow as tf
import sox


def read_wavfile(path, **librosa_kwargs):
    try:
        return librosa.core.load(path, **librosa_kwargs)
    except EOFError:
        return None, 0

def write_wav(wav, path):
    signal, rate = wav
    librosa.output.write_wav(path, signal, rate)

def get_samplerate(path, **librosa_kwargs):
    return librosa.core.get_samplerate(path, **librosa_kwargs)

def get_audio_type(path):
    return sox.file_info.file_type(path)

def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def write_features(sequence_features, target_path, features_meta):
    target_path += ".tfrecord"
    with tf.io.TFRecordWriter(target_path, options="GZIP") as record_writer:
        for sequence in sequence_features:
            sequence_example = sequence_to_example(*sequence)
            record_writer.write(sequence_example.SerializeToString())
    with open(target_path + ".meta.json", "w") as meta_file:
        json.dump(features_meta, meta_file)
    return target_path

def load_features_as_dataset(tfrecord_paths, model_config):
    with open(tfrecord_paths[0] + ".meta.json") as f:
        features_meta = json.load(f)
    dataset = tf.data.TFRecordDataset(tfrecord_paths, compression_type="GZIP")
    #TODO set data dimensions elegantly from somewhere
    dataset = dataset.map(lambda se: sequence_example_to_model_input(se, features_meta["num_labels"], features_meta["num_features"]))
    if model_config.get("dataset_shuffle_size", 0):
        dataset = dataset.shuffle(model_config["dataset_shuffle_size"])
    dataset = dataset.repeat()
    dataset = dataset.batch(model_config["batch_size"])
    return dataset, features_meta

def write_utterance(utterance, basedir):
    lang_label, (wav, rate) = utterance
    filename = hashlib.md5(bytes(wav)).hexdigest() + '.npy'
    with open(os.path.join(basedir, filename), "wb") as out_file:
        np.save(out_file, (lang_label, (wav, rate)), allow_pickle=True, fix_imports=False)

def load_utterance(path):
    with open(path, "rb") as np_file:
        data = np.load(np_file, allow_pickle=True, fix_imports=False)
        return data[0], (data[1][0], data[1][1])

def load_utterances(basedir):
    for path in os.listdir(basedir):
        yield load_utterance(os.path.join(basedir, path))

def count_dataset(tfrecord_paths):
    """
    Count the amount of entries in a TFRecord file by iterating over it once.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_paths, compression_type="GZIP")
    next_element = tf.data.make_one_shot_iterator(dataset).get_next()
    num_elements = 0
    with tf.Session() as session:
        try:
            while True:
                session.run(next_element)
                num_elements += 1
        except tf.errors.OutOfRangeError:
            # Iterator exhausted
            pass
    return num_elements
