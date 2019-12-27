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
def framewise_energy_vad_decisions(signals, frame_length=400, frame_step=160, strength=0.5, min_rms_threshold=1e-3):
    """
    For a batch of 1D-signals, compute energy based frame-wise VAD decisions by comparing the RMS value of each frame to the mean RMS of the whole signal (separately for each signal), such that True means the frame is voiced and False unvoiced.
    VAD threshold is 'strength' multiplied by mean RMS, i.e. larger 'strength' values increase VAD aggressiveness.
    """
    tf.debugging.assert_rank(signals, 2, message="energy_vad_decisions expects batches of single channel signals")
    frames = tf.signal.frame(signals, frame_length, frame_step)
    rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.abs(frames)), axis=2))
    mean_rms = tf.math.reduce_mean(rms, axis=1, keepdims=True)
    threshold = strength * tf.math.maximum(min_rms_threshold, mean_rms)
    return tf.math.greater(rms, threshold)
