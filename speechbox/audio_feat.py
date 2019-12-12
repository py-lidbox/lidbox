"""
Audio feature extraction.
Some functions are simply one-to-one TensorFlow math conversions from https://github.com/librosa.
"""
import tensorflow as tf


@tf.function
def fft_frequencies(sample_rate=16000, n_fft=400):
    # Equal to librosa.core.fft_frequencies
    begin = 0.0
    end = tf.cast(sample_rate // 2, tf.float32)
    step = 1 + n_fft // 2
    return tf.linspace(begin, end, step)

# All functions (except energy_vad) operate on batches, input must always have ndim of 2.
# E.g. if the input is a single signal of length N, use a singleton batch of shape [1, N].

@tf.function
def power_to_db(S, ref=tf.math.reduce_max, amin=1e-10, top_db=80.0):
    e = tf.math.exp(1.0)
    log_spec = e * (tf.math.log(tf.math.maximum(amin, S)) - tf.math.log(tf.math.maximum(amin, ref(S))))
    return tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)

@tf.function
def spectrograms(signals, sample_rate=16000, frame_length=400, frame_step=160, power=2.0, fmin=0.0, fmax=8000.0):
    tf.debugging.assert_rank(signals, 2, "Expected input signals from which to compute spectrograms to be of shape (batch_size, signal_frames)")
    # This should be more or less the same as:
    # S = np.power(np.abs(librosa.core.stft(y, n_fft=400, hop_length=160, center=False)), 2)
    S = tf.signal.stft(signals, frame_length, frame_step, fft_length=frame_length)
    S = tf.math.pow(tf.math.abs(S), power)
    # Drop all fft bins that are outside the given [fmin, fmax] band
    fft_freqs = fft_frequencies(sample_rate=sample_rate, n_fft=frame_length)
    # With default [fmin, fmax] of [0, 8000] this will contain only 'True's
    bins_in_band = tf.math.logical_and(
        tf.math.greater_equal(fft_freqs, fmin),
        tf.math.less_equal(fft_freqs, fmax)
    )
    return tf.boolean_mask(S, bins_in_band, axis=2)

@tf.function
def melspectrograms(S, sample_rate=16000, num_mel_bins=40, fmin=60.0, fmax=6000.0):
    tf.debugging.assert_rank(S, 3, "Input to melspectrograms must be a batch of 2-dimensional spectrograms with shape (batch, frames, freq_bins)")
    tf.debugging.assert_greater(tf.shape(S)[2], 0, "Expected given spectrogram to have a positive amount of frequency bins")
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=tf.shape(S)[2],
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )
    return tf.matmul(S, mel_weights)

@tf.function
def energy_vad(signal, frame_length=400, strength=0.5, min_rms_threshold=1e-3):
    """
    Perform frame-wise vad decisions based on mean RMS value for each frame in a given 'signal'.
    VAD threshold is 'strength' multiplied by mean RMS (larger 'strength' values increase VAD aggressiveness and drops more frames).
    """
    tf.debugging.assert_rank(signal, 1, "energy_vad supports VAD only on one signal at a time, e.g. not batches")
    frames = tf.signal.frame(signal, frame_length, frame_length)
    rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.abs(frames)), axis=1))
    mean_rms = tf.math.reduce_mean(rms, axis=0, keepdims=True)
    threshold = strength * tf.math.maximum(min_rms_threshold, mean_rms)
    # Take only frames that have rms greater than the threshold
    filtered_frames = tf.boolean_mask(frames, tf.math.greater(rms, threshold))
    # Concat all frames and return a new signal with silence removed
    return tf.reshape(filtered_frames, [-1])
