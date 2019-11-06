"""
Some functions from https://github.com/librosa converted to TensorFlow math.
"""
import io

import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf


@tf.function
def power_to_db(S, ref=tf.math.reduce_max, amin=1e-10, top_db=80.0):
    e = tf.math.exp(1.0)
    log_spec = e * (tf.math.log(tf.math.maximum(amin, S)) - tf.math.log(tf.math.maximum(amin, ref(S))))
    return tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)

@tf.function
def spectrograms(signals, n_fft=400, hop_length=160, power=2.0):
    S = tf.math.abs(tf.signal.stft(signals, n_fft, hop_length, fft_length=n_fft))
    return tf.math.pow(S, power)

@tf.function
def melspectrograms(S, sample_rate, num_mel_bins=40, fmin=60.0, fmax=6000.0):
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=S.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )
    return tf.matmul(S, mel_weights)

@tf.function
def mfcc(melspectrogram, n_coefs=13):
    log_mel_spectrogram = tf.math.log(melspectrogram + 1e-6)
    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., 1:n_coefs]

@tf.function
def energy_vad(signals, n_fft=400, hop_length=160, vad_strength=0.7):
    """Returns frame-wise vad decisions."""
    tf.debugging.assert_non_negative(vad_strength)
    frames = tf.signal.frame(signals, n_fft, hop_length)
    rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.abs(frames)), axis=2))
    threshold = vad_strength * 0.5 * tf.math.reduce_mean(rms, axis=1, keepdims=True)
    return tf.math.greater(rms, threshold)

@tf.function
def extract_melspec_features(signals, sample_rate):
    melspecs = melspectrograms(spectrograms(signals), sample_rate)
    vad_decisions = energy_vad(signals)
    return tf.ragged.boolean_mask(melspecs, vad_decisions)

def get_heatmap_plot(batch):
    figure = plt.figure(figsize=(10, 10))
    width = 5
    height = batch.shape[1] // width + (batch.shape[1] % width) != 0
    for i, data in enumerate(batch, start=1):
        ax = plt.subplot(height, width, i)
        ax.set_title(str(i))
        ax.invert_yaxis()
        seaborn.heatmap(data.T, ax=ax, cmap="YlGnBu_r")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    figure.close()
    buf.seek(0)
    heatmaps_png = tf.io.decode_png(buf.getvalue(), channels=4)
    return heatmaps_png

def get_spectrogram_plot_from_signal(signals, sample_rate):
    melspecs_db = power_to_db(melspectrogram(signals, sample_rate))
    return get_heatmap_plot(melspecs_db)
