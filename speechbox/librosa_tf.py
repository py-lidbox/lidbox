"""
Some functions from https://github.com/librosa converted to TensorFlow math.
"""
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
