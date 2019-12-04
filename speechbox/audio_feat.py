"""
Audio feature extraction.
Some functions are simply one-to-one TensorFlow math conversions from https://github.com/librosa.
"""
from tensorflow import (
    abs,
    exp,
    greater,
    maximum,
    minimum,
    pow,
    reduce_max,
    reduce_mean,
    reduce_min,
    sqrt,
    square,
)
import tensorflow as tf


# All functions operate on batches, input must always have ndim of 2.
# E.g. if the input is a single signal of length N, use a singleton batch of shape [1, N].
# Also note that we are using tf.math, not the Python standard library.

@tf.function
def power_to_db(S, ref=tf.math.reduce_max, amin=1e-10, top_db=80.0):
    log_spec = exp(1.0) * (log(maximum(amin, S)) - log(maximum(amin, ref(S))))
    return maximum(log_spec, reduce_max(log_spec) - top_db)

@tf.function
def spectrograms(signals, frame_length=512, frame_step=160, power=2.0):
    S = abs(tf.signal.stft(signals, frame_length, frame_step))
    return pow(S, power)

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
def energy_vad(signal, frame_length=512, strength=0.3, min_rms_threshold=1e-3):
    """
    Perform frame-wise vad decisions based on mean RMS value for each frame in a given 'signal'.
    VAD threshold is 'strength' multiplied by mean RMS (larger 'strength' values increase VAD aggressiveness and drops more frames).
    """
    tf.debugging.assert_greater(frame_length, 0, "energy_vad requires a non zero frame_length to do VAD on")
    tf.debugging.assert_rank(signal, 1, "energy_vad supports VAD only on one signal at a time, e.g. not batches")
    frames = tf.signal.frame(signal, frame_length, frame_length)
    rms = sqrt(reduce_mean(square(abs(frames)), axis=1))
    mean_rms = reduce_mean(rms, axis=0, keepdims=True)
    threshold = strength * maximum(min_rms_threshold, mean_rms)
    # Take only frames that have rms greater than the threshold
    filtered_frames = tf.boolean_mask(frames, greater(rms, threshold))
    # Concat all frames and return a new signal with silence removed
    return tf.reshape(filtered_frames, [-1])
