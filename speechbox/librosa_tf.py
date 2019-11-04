"""
Some functions from https://github.com/librosa converted to TensorFlow math.
"""
import tensorflow as tf

@tf.function
def power_to_db(S, ref=tf.math.reduce_max, amin=1e-10, top_db=80.0):
    e = tf.math.exp(1.0)
    log_spec = e * (tf.math.log(tf.math.maximum(amin, S)) - tf.math.log(tf.math.maximum(amin, ref(S))))
    return tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)

@tf.function
def spectrogram(signal, n_fft=2048, hop_length=512, power=1.0):
    S = tf.math.abs(tf.signal.stft(signal, n_fft, hop_length, fft_length=n_fft))
    return tf.math.pow(S, power)

@tf.function
def melspectrogram(signal, sample_rate, num_mel_bins=128):
    S = spectrogram(signal)
    W_mel = tf.signal.linear_to_mel_weight_matrix(sample_rate=sample_rate, num_spectrogram_bins=S.shape[1], num_mel_bins=num_mel_bins)
    print(S.shape, W_mel.shape)
    return power_to_db(tf.matmul(S, W_mel))
