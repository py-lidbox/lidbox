"""
Audio feature extraction.
Some functions are simply one-to-one TensorFlow math conversions from https://github.com/librosa.
"""
import collections

import tensorflow as tf
import numpy as np
import webrtcvad


Wav = collections.namedtuple("Wav", ["audio", "sample_rate"])

@tf.function
def fft_frequencies(sample_rate=16000, n_fft=400):
    # Equal to librosa.core.fft_frequencies
    begin = 0.0
    end = tf.cast(sample_rate // 2, tf.float32)
    step = 1 + n_fft // 2
    return tf.linspace(begin, end, step)

@tf.function
def log10(x):
    return tf.math.log(x) / tf.math.log(10.0)

@tf.function
def power_to_db(S, ref=tf.math.reduce_max, amin=1e-10, top_db=80.0):
    db_spectrogram = 10.0 * (log10(tf.math.maximum(amin, S)) - log10(tf.math.maximum(amin, ref(S))))
    return tf.math.maximum(db_spectrogram, tf.math.reduce_max(db_spectrogram) - top_db)

@tf.function
def spectrograms(signals, frame_length=400, frame_step=160, power=2.0, fmin=0.0, fmax=8000.0):
    tf.debugging.assert_rank(signals.audio, 2, "Expected input signals from which to compute spectrograms to be of shape (batch_size, signal_frames)")
    # This should be more or less the same as:
    # S = np.power(np.abs(librosa.core.stft(y, n_fft=400, hop_length=160, center=False)), 2)
    # S = tf.signal.stft(signals.audio, frame_length, frame_step, fft_length=frame_length)
    S = tf.signal.stft(signals.audio, frame_length, frame_step, fft_length=frame_length)
    S = tf.math.pow(tf.math.abs(S), power)
    # Drop all fft bins that are outside the given [fmin, fmax] band
    fft_freqs = fft_frequencies(sample_rate=signals.sample_rate[0], n_fft=frame_length)
    # With default [fmin, fmax] of [0, 8000] this will contain only 'True's
    bins_in_band = tf.math.logical_and(fmin <= fft_freqs, fft_freqs <= fmax)
    return tf.boolean_mask(S, bins_in_band, axis=2)

@tf.function
def melspectrograms(S, sample_rate, num_mel_bins=40, fmin=60.0, fmax=6000.0):
    tf.debugging.assert_rank(S, 3, "Input to melspectrograms must be a batch of 2-dimensional spectrograms with shape (batch, frames, freq_bins)")
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=tf.shape(S)[2],
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax)
    return tf.matmul(S, mel_weights)

@tf.function
def framewise_rms_energy_vad_decisions(signals, frame_length=400, frame_step=160, strength=0.5, min_rms_threshold=1e-3):
    """
    For a batch of 1D-signals, compute energy based frame-wise VAD decisions by comparing the RMS value of each frame to the mean RMS of the whole signal (separately for each signal), such that True means the frame is voiced and False unvoiced.
    VAD threshold is 'strength' multiplied by mean RMS, i.e. larger 'strength' values increase VAD aggressiveness.
    """
    tf.debugging.assert_rank(signals, 2, message="energy_vad_decisions expects batches of single channel signals")
    frames = tf.signal.frame(signals, frame_length, frame_step, axis=1)
    rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.abs(frames)), axis=2))
    mean_rms = tf.math.reduce_mean(rms, axis=1, keepdims=True)
    threshold = strength * tf.math.maximum(min_rms_threshold, mean_rms)
    return rms > threshold

# similar to kaldi mfcc vad but without a context window (for now):
# https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/src/ivector/voice-activity-detection.cc
# TODO something is wrong, this drops almost all frames
@tf.function
def framewise_mfcc_energy_vad_decisions(wav, spec_kwargs, melspec_kwargs, energy_threshold=5.0, energy_mean_scale=0.5):
    tf.debugging.assert_rank(wav.audio, 1, message="Expected a single 1D signal (i.e. one mono audio tensor without explicit channels)")
    S = spectrograms(Wav(tf.expand_dims(wav.audio, 0), tf.expand_dims(wav.sample_rate, 0)), **spec_kwargs)
    S = melspectrograms(S, sample_rate=wav.sample_rate, **melspec_kwargs)
    S = tf.math.log(S + 1e-6)
    tf.debugging.assert_all_finite(S, "logmelspectrogram extraction failed, cannot compute mfcc energy vad")
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(S)
    log_energy = tf.squeeze(tf.signal.mfccs_from_log_mel_spectrograms(S)[..., 0])
    tf.debugging.assert_rank(log_energy, 1, message="Failed to extract 0th MFCC coef")
    mean_log_energy = tf.math.reduce_mean(log_energy)
    vad_decisions = log_energy > (energy_threshold + energy_mean_scale * mean_log_energy)
    return vad_decisions

@tf.function
def wav_to_bytes(wav):
    tf.debugging.assert_rank(wav.audio, 1, message="Expected a single 1D signal (i.e. one mono audio tensor without explicit channels)")
    wav_bytes = tf.audio.encode_wav(tf.expand_dims(wav.audio, -1), wav.sample_rate)
    # expecting one wav pcm header and sample width of 2
    tf.debugging.assert_equal(tf.strings.length(wav_bytes) - 44, 2 * tf.size(wav.audio), message="wav encoding failed")
    # drop wav header
    wav_bytes = tf.strings.substr(wav_bytes, 44, -1)
    return wav_bytes

#TODO cleanup and/or webrtcvad in tf graph (might be nontrivial)
def framewise_webrtcvad_decisions(wav_length, wav_bytes, sample_rate, vad_frame_length, vad_frame_step, feat_frame_length, feat_frame_step, aggressiveness):
    frame_begin_pos = 2 * np.arange(0, wav_length, feat_frame_step)
    num_feat_frames = (wav_length - feat_frame_length + feat_frame_step) // feat_frame_step
    num_vad_frames = (feat_frame_length - vad_frame_length + vad_frame_step) // vad_frame_step
    frame_begin_pos = frame_begin_pos[:num_feat_frames]
    vad_decisions = np.zeros([num_feat_frames], np.bool)
    vad = webrtcvad.Vad(aggressiveness)
    for i, feat_begin in enumerate(frame_begin_pos):
        feat_frame_bytes = wav_bytes[feat_begin:feat_begin + 2 * feat_frame_length]
        vad_begin = 2 * np.arange(0, feat_frame_length - vad_frame_step, vad_frame_step)
        for j in vad_begin:
            vad_frame = feat_frame_bytes[j:j + 2 * vad_frame_length]
            vad_decisions[i] |= (len(vad_frame) < 2 * vad_frame_length or vad.is_speech(vad_frame, sample_rate))
    return vad_decisions
