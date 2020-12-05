"""
Copied from
https://github.com/tensorflow/tensorflow/blob/fab3f8548264590133d2f49c75ed9c0c0ab83f28/tensorflow/python/ops/signal/mel_ops.py

Original implementation does not play well with tf.function and tfjs-converter.
"""
import tensorflow as tf


# tfjs-converter cannot convert tf.linspace due to broadcasting issues
def _linspace(start, stop, num, dtype=tf.float32):
    range = tf.range(0, num, dtype=dtype)
    start = tf.cast(start, dtype)
    stop = tf.cast(stop, dtype)
    num = tf.cast(num, dtype)
    return start + (stop - start) * range / num


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _hertz_to_mel(frequencies_hertz, name=None):
    return _MEL_HIGH_FREQUENCY_Q * tf.math.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=8000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0,
                                dtype=tf.float32,
                                name=None):
    zero = tf.constant(0.0, dtype)

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = tf.cast(sample_rate, dtype) / 2.0
    linear_frequencies = _linspace(
        zero, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    spectrogram_bins_mel = tf.expand_dims(
        _hertz_to_mel(linear_frequencies), 1)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = tf.signal.frame(
        _linspace(
            _hertz_to_mel(lower_edge_hertz),
            _hertz_to_mel(upper_edge_hertz),
            num_mel_bins + 2),
        frame_length=3,
        frame_step=1)

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(tf.reshape(
        t, [1, num_mel_bins]) for t in tf.split(
            band_edges_mel, 3, axis=1))

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = tf.maximum(
        zero, tf.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    return tf.pad(
        mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], name=name)
