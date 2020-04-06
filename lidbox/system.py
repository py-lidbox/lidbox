"""
Misc. IO stuff.
"""
import hashlib
import subprocess


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
    from scipy.io import arff
    import numpy as np
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

def iter_log_events(tf_event_file):
    import tensorflow as tf
    from tensorflow.core.util.event_pb2 import Event
    for event in tf.data.TFRecordDataset([tf_event_file]):
        event = Event.FromString(event.numpy())
        if event.summary.value:
            assert len(event.summary.value) == 1, "Unexpected length for event summary"
            value = event.summary.value[0]
            yield value.tag, value.simple_value

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
