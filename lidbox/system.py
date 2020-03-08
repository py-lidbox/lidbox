"""File IO."""
import gzip
import hashlib
import json
import os
import subprocess

from scipy.io import arff
import librosa
import numpy as np
import sox
import yaml


SUBPROCESS_BATCH_SIZE = 5000

def run_command(cmd):
    process = subprocess.run(
        cmd.split(" "),
        check=True,
        stdout=subprocess.PIPE
    )
    return process.stdout.decode("utf-8").rstrip()

def run_for_files(cmd, filepaths, batch_size=SUBPROCESS_BATCH_SIZE):
    # Run in batches
    for begin in range(0, len(filepaths), batch_size):
        batch = ' '.join(filepaths[begin:begin+batch_size])
        yield run_command(cmd + ' ' + batch)

def read_arff_features(path, include_keys=None, exclude_keys=None, types=None):
    if types is None:
        types = {"numeric"}
    if exclude_keys is None:
        exclude_keys = {"frameTime"}
    data, meta = arff.loadarff(path)
    keys = [
        key for key, type in zip(meta.names(), meta.types())
        if (include_keys is None or key in include_keys) and key not in exclude_keys and type in types
    ]
    assert all(data[key].shape == data[keys[0]].shape for key in keys), "inconsistent dimensions in arff file, expected all to have shape {}".format(data[keys[0]].shape)
    feats = np.vstack([data[key] for key in keys if not np.any(np.isnan(data[key]))])
    return feats.T, keys

def write_wav(path, wav):
    signal, rate = wav
    librosa.output.write_wav(path, signal, rate)

def get_samplerate(path, **librosa_kwargs):
    return librosa.core.get_samplerate(path, **librosa_kwargs)

def get_audio_type(path):
    try:
        return sox.file_info.file_type(path)
    except sox.core.SoxiError:
        return None

def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def all_md5sums(paths, num_workers=32):
    from multiprocessing import Pool
    with Pool(num_workers) as pool:
        return pool.map(md5sum, paths)

def load_gzip_json(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        return json.load(f)

def dump_gzip_json(data, path):
    with gzip.open(path, "wb") as f:
        json_str = json.dumps(data, sort_keys=True, indent=2)
        f.write(json_str.encode("utf-8"))

def append_json(data, path):
    if os.path.exists(path):
        with open(path) as f:
            data_list = json.load(f)
    else:
        data_list = []
    data_list.append(data)
    with open(path, "w") as f:
        json.dump(data_list, f)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def write_utterance(utterance, basedir):
    label, (wav, rate) = utterance
    filename = hashlib.md5(bytes(wav)).hexdigest() + '.npy'
    with open(os.path.join(basedir, filename), "wb") as out_file:
        np.save(out_file, (label, (wav, rate)), allow_pickle=True, fix_imports=False)

def load_utterance(path):
    with open(path, "rb") as np_file:
        data = np.load(np_file, allow_pickle=True, fix_imports=False)
        return data[0], (data[1][0], data[1][1])

def load_utterances(basedir):
    for path in os.listdir(basedir):
        yield load_utterance(os.path.join(basedir, path))

def load_audiofile_paths(pathlist_file):
    with open(pathlist_file) as f:
        for line in f:
            split = line.split()
            wavpath, rest = split[0].strip(), split[1:]
            wav, _ = read_wavfile(wavpath)
            if wav is not None:
                yield wavpath, rest

def concatenate_wavs(wavs):
    assert len(wavs) > 0, "Nothing to concatenate"
    assert all(rate == wavs[0][1] for _, rate in wavs), "Cannot concatenate wavfiles with different sampling rates"
    rate = wavs[0][1]
    return np.concatenate([wav for wav, _ in wavs]), rate

def get_most_recent_file(directory):
    # Get path object with greatest unix timestamp
    files = (f for f in os.scandir(directory) if f.is_file())
    return max(files, key=lambda d: d.stat().st_mtime).name

def count_all_features(features_file):
    from tensorflow import device
    with device("/CPU:0"):
        dataset, meta = load_features_as_dataset([features_file])
        return int(dataset.reduce(0, lambda count, _: count + 1)), meta

def count_all_features_parallel(labels, features_files, num_workers=None):
    from multiprocessing import Pool
    assert len(labels) == len(features_files)
    if num_workers is None:
        num_workers = len(features_files)
    with Pool(num_workers) as pool:
        return zip(labels, pool.map(count_all_features, features_files))

def load_features_meta(tfrecord_path):
    with open(tfrecord_path + ".meta.json") as f:
        return json.load(f)

def iter_log_events(tf_event_file):
    import tensorflow as tf
    from tensorflow.core.util.event_pb2 import Event
    for event in tf.data.TFRecordDataset([tf_event_file]):
        event = Event.FromString(event.numpy())
        if event.summary.value:
            assert len(event.summary.value) == 1, "Unexpected length for event summary"
            value = event.summary.value[0]
            yield value.tag, value.simple_value

def apply_sox_transformer(src_paths, dst_paths, transform_steps):
    t = sox.Transformer()
    for transform, value in transform_steps:
        if transform == "normalize":
            t = t.norm(float(value))
        elif transform == "volume":
            t = t.vol(float(value), gain_type="amplitude")
        elif transform == "speed":
            t = t.speed(float(value))
        elif transform == "reverse" and value:
            t = t.reverse()
    # Try to apply the transformation on every src_path, building output files into every dst_path
    for src, dst in zip(src_paths, dst_paths):
        if t.build(src, dst):
            yield src, dst
        else:
            yield src, None

def get_total_duration_sec(paths):
    # Run SoXi for all files
    soxi_cmd = "soxi -D -T"
    seconds = sum(float(output) for output in run_for_files(soxi_cmd, paths))
    return round(seconds)

def get_total_duration(paths):
    secs = get_total_duration_sec(paths)
    mins, secs = secs // 60, secs % 60
    hours, mins = mins // 60, mins % 60
    return hours, mins, secs

def format_duration(duration):
    return "{:02d}h {:02d}min {:02d}sec".format(*duration)

def parse_path_list(path):
    paths = []
    labels = []
    with open(path) as f:
        for line in f:
            path, label = line.strip().split()[:2]
            paths.append(path)
            labels.append(label)
    return paths, labels

def config_checksum(config, datagroup_key):
    md5input = {k: config[k] for k in ("features", "datasets")}
    json_str = json.dumps(md5input, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    return json_str, hashlib.md5(json_str.encode("utf-8")).hexdigest()
