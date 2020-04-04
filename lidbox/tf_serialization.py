import os

import tensorflow as tf

import lidbox
if lidbox.DEBUG:
    # tf.autograph.set_verbosity(10, alsologtostdout=True)
    TF_AUTOTUNE = None
else:
    TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

from . import audio_feat


def floats2floatlist(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def int2int64list(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def sequence2floatlists(v_seq):
    return tf.train.FeatureList(feature=(floats2floatlist(v) for v in v_seq))

def string2byteslist(s):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s]))

def serialize_sample(features, meta, wav_audio, wav_sample_rate):
    context_definition = {
        "uuid": string2byteslist(meta[0]),
        "label": string2byteslist(meta[1]),
        "wav_audio": floats2floatlist(wav_audio),
        "wav_sample_rate": int2int64list(wav_sample_rate),
    }
    context = tf.train.Features(feature=context_definition)
    sequence_definition = {
        "features": sequence2floatlists(features),
    }
    feature_lists = tf.train.FeatureLists(feature_list=sequence_definition)
    seq_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return seq_example.SerializeToString()

def deserialize_sample(seq_example_str, feature_dim, wav_length):
    context_definition = {
        "uuid": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
        "wav_audio": tf.io.FixedLenFeature([wav_length], tf.float32),
        "wav_sample_rate": tf.io.FixedLenFeature([], tf.int64),
    }
    sequence_definition = {
        "features": tf.io.FixedLenSequenceFeature([feature_dim], tf.float32)
    }
    context, sequence = tf.io.parse_single_sequence_example(
        seq_example_str,
        context_features=context_definition,
        sequence_features=sequence_definition
    )
    meta = (
        context["uuid"],
        context["label"],
        audio_feat.Wav(
            context["wav_audio"],
            tf.cast(context["wav_sample_rate"], tf.int32)))
    return sequence["features"], meta

def write_features(extractor_ds, tfrecord_dir, max_group_size=int(1e3)):
    def get_group_index(index, _):
        return index // max_group_size
    def write_group(group_index, group):
        tfrecord_path = tf.strings.join([
            tfrecord_dir,
            os.sep,
            tf.strings.as_string(group_index, width=6, fill='0'),
            ".tfrecord"])
        writer = tf.data.experimental.TFRecordWriter(tfrecord_path, compression_type="GZIP")
        drop_enumeration = lambda _, x: x
        writer.write(group.map(drop_enumeration))
        return tf.data.Dataset.from_tensors(tfrecord_path)
    def serialize(feats, meta):
        args = feats, (meta[0], meta[1]), meta[2].audio, meta[2].sample_rate
        return tf.numpy_function(serialize_sample, args, tf.string)
    group_writer = tf.data.experimental.group_by_window(
            get_group_index,
            write_group,
            window_size=max_group_size)
    return (extractor_ds
            .map(serialize, num_parallel_calls=TF_AUTOTUNE)
            .enumerate()
            .apply(group_writer))

def load_features(tfrecord_dir, feature_dim, wav_length):
    def deserialize(sample_str):
        return deserialize_sample(sample_str, feature_dim, wav_length)
    tfrecord_paths = [p.path for p in os.scandir(tfrecord_dir) if p.name.endswith(".tfrecord")]
    return (tf.data.TFRecordDataset(
                tfrecord_paths,
                compression_type="GZIP",
                buffer_size=100*int(1e6),
                num_parallel_reads=TF_AUTOTUNE)
            .map(deserialize, num_parallel_calls=TF_AUTOTUNE))
