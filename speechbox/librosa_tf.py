"""
Some functions from https://github.com/librosa converted to TensorFlow math.
"""
import io

import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf


# All functions operate on batches, input must always have ndim of 2.
# i.e. use a singleton batch for single input

@tf.function
def power_to_db(S, ref=tf.math.reduce_max, amin=1e-10, top_db=80.0):
    e = tf.math.exp(1.0)
    log_spec = e * (tf.math.log(tf.math.maximum(amin, S)) - tf.math.log(tf.math.maximum(amin, ref(S))))
    return tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)

@tf.function
def spectrograms(signals, frame_length=400, frame_step=160, power=1.0):
    S = tf.math.abs(tf.signal.stft(signals, frame_length, frame_step))
    return tf.math.pow(S, power)

@tf.function
def melspectrograms(S, sample_rate=16000, num_mel_bins=40, fmin=60.0, fmax=6000.0):
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=S.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )
    return tf.matmul(S, mel_weights)

@tf.function
def energy_vad(signals, frame_length=400, frame_step=160, strength=0.7, min_rms_threshold=1e-3):
    """Returns frame-wise vad decisions."""
    tf.debugging.assert_non_negative(strength)
    frames = tf.signal.frame(signals, frame_length, frame_step)
    rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.abs(frames)), axis=2))
    mean_rms = tf.math.reduce_mean(rms, axis=1, keepdims=True)
    threshold = strength * 0.5 * tf.math.maximum(min_rms_threshold, mean_rms)
    return tf.math.greater(rms, threshold)

@tf.function
def extract_features_and_do_vad(signals, feattype, spec_kwargs, vad_kwargs, melspec_kwargs, logmel, mfcc_kwargs):
    feat = spectrograms(signals, **spec_kwargs)
    if feattype in ("melspectrogram", "logmelspectrogram", "mfcc"):
        feat = melspectrograms(feat, **melspec_kwargs)
        if feattype in ("logmelspectrogram", "mfcc"):
            feat = tf.math.log(feat + 1e-6)
            if feattype == "mfcc":
                num_coefs = mfcc_kwargs.get("num_coefs", 13)
                feat = tf.signal.mfccs_from_log_mel_spectrograms(feat)[..., 1:num_coefs+1]
    vad_decisions = energy_vad(signals, **vad_kwargs)
    # We use ragged tensors here to keep the dimensions after filtering feature frames according to vad decision based on the signals batch
    return tf.ragged.boolean_mask(feat, vad_decisions)

def get_heatmap_plot(seq_examples, num_rows, num_cols):
    decode = lambda t: t.numpy().decode("utf-8")
    figure = plt.figure(figsize=(30, 20))
    for i, (data, meta) in enumerate(seq_examples, start=1):
        ax = plt.subplot(num_rows, num_cols, i)
        ax.set_title(decode(meta[0]) + ": " + decode(meta[1]))
        seaborn.heatmap(data.numpy().T, ax=ax, cmap="YlGnBu_r")
        ax.invert_yaxis()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    heatmaps_png = tf.io.decode_png(buf.getvalue(), channels=4)
    return buf.getvalue()

def get_spectrogram_plot_from_signal(signals, sample_rate):
    melspecs_db = power_to_db(melspectrogram(signals, sample_rate))
    return get_heatmap_plot(melspecs_db)
