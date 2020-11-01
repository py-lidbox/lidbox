"""
Audio feature extraction.
Many functions have been inspired by https://github.com/librosa and https://github.com/kaldi-asr/kaldi.
"""
import os
import wave

import miniaudio
import numpy as np
import tensorflow as tf
import webrtcvad


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
def read_wav(path):
    wav_bytes = tf.io.read_file(path)
    audio = tf.audio.decode_wav(wav_bytes)
    # Merge channels by averaging, for mono this just drops the channel dim.
    signal = tf.math.reduce_mean(audio.audio, axis=1, keepdims=False)
    return signal, audio.sample_rate


def miniaudio_read_mp3(path):
    audio = miniaudio.mp3_read_file_f32(path.decode("utf-8"))
    return np.array(audio.samples, np.float32), audio.sample_rate

@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
def read_mp3(path):
    signal, rate = tf.numpy_function(miniaudio_read_mp3, [path], [tf.float32, tf.int64])
    return signal, tf.cast(rate, tf.int32)


@tf.function
def write_mono_wav(path, signal, sample_rate):
    tf.debugging.assert_rank(signal, 1, "write_wav expects 1-dim mono signals without channel dims.")
    signal = tf.expand_dims(signal, -1)
    wav = tf.audio.encode_wav(signal, sample_rate)
    tf.io.write_file(path, wav)
    return path


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)])
def wav_to_pcm_data(signal, sample_rate):
    pcm_data = tf.audio.encode_wav(tf.expand_dims(signal, -1), sample_rate)
    # expecting one wav pcm header and sample width of 2
    tf.debugging.assert_equal(tf.strings.length(pcm_data) - 44, 2 * tf.size(signal), message="wav encoding failed")
    header = tf.strings.substr(pcm_data, 0, 44)
    content = tf.strings.substr(pcm_data, 44, -1)
    return header, content


def numpy_snr_mixer(clean, noise, snr):
    """
    Mix signal 'noise' into signal 'clean' at given SNR.
    From
    https://github.com/microsoft/MS-SNSD/blob/e84aba38cac499a109c0d237a00dc600dcf9b7e7/audiolib.py
    """
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5
    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech


@tf.function
def snr_mixer(clean, noise, snr):
    """
    TensorFlow version of numpy_snr_mixer.
    """
    tf.debugging.assert_equal(tf.size(clean), tf.size(noise), message="mismatching length for signals clean and noise given to snr mixer")
    # Normalizing to -25 dB FS
    scalarclean = tf.math.pow(10.0, -25.0/20.0) / root_mean_square(clean)
    clean_norm = scalarclean * clean
    rmsclean = root_mean_square(clean_norm)
    scalarnoise = tf.math.pow(10.0, -25.0/20.0) / root_mean_square(noise)
    noise_norm = scalarnoise * noise
    rmsnoise = root_mean_square(noise_norm)
    # Set the noise level for a given SNR
    level = tf.math.pow(10.0, snr / 20.0)
    noisescalar = tf.math.sqrt(rmsclean / level / rmsnoise)
    noisenewlevel = noisescalar * noise_norm
    noisyspeech = clean_norm + noisenewlevel
    return clean_norm, noisenewlevel, noisyspeech


@tf.function
def fft_frequencies(sample_rate, n_fft):
    # Equal to librosa.core.fft_frequencies
    begin = 0.0
    end = tf.cast(sample_rate // 2, tf.float32)
    step = 1 + n_fft // 2
    return tf.linspace(begin, end, step)


@tf.function
def log10(x):
    return tf.math.log(x) / tf.math.log(10.0)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32)])
def power_to_db(S, amin=1e-10, top_db=80.0):
    db_spectrogram = 20.0 * (log10(tf.math.maximum(amin, S)) - log10(tf.math.maximum(amin, tf.math.reduce_max(S))))
    return tf.math.maximum(db_spectrogram, tf.math.reduce_max(db_spectrogram) - top_db)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.int32)])
def ms_to_frames(sample_rate, ms):
    return tf.cast(tf.cast(sample_rate, tf.float32) * 1e-3 * tf.cast(ms, tf.float32), tf.int32)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)])
def spectrograms(signals, sample_rate, frame_length_ms=25, frame_step_ms=10, power=2.0, fmin=0.0, fmax=8000.0, fft_length=512):
    frame_length = ms_to_frames(sample_rate, frame_length_ms)
    frame_step = ms_to_frames(sample_rate, frame_step_ms)
    S = tf.signal.stft(signals, frame_length, frame_step, fft_length=fft_length)
    S = tf.math.pow(tf.math.abs(S), power)
    # Drop all fft bins that are outside the given [fmin, fmax] band
    fft_freqs = fft_frequencies(sample_rate=sample_rate, n_fft=fft_length)
    # With default [fmin, fmax] of [0, 8000] this will contain only 'True's
    bins_in_band = tf.math.logical_and(fmin <= fft_freqs, fft_freqs <= fmax)
    return tf.boolean_mask(S, bins_in_band, axis=2)


def linear_to_mel(S, sample_rate, num_mel_bins=40, fmin=0.0, fmax=8000.0):
    tf.debugging.assert_rank(S, 3, message="Expected a batch of spectrograms")
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=tf.shape(S)[2],
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax)
    return tf.tensordot(S, mel_weights, 1)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)])
def root_mean_square(x, axis=-1):
    square = tf.math.square(tf.math.abs(x))
    mean = tf.math.reduce_mean(square, axis=axis)
    root = tf.math.sqrt(mean)
    return root


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.int32)])
def run_length_encoding(v):
    """
    From numpy version at https://stackoverflow.com/a/32681075
    """
    i = tf.concat(([-1], tf.squeeze(tf.where(v[1:] != v[:-1]), axis=1), [tf.size(v) - 1]), axis=0, name="rle_concat_indices")
    pos = tf.concat(([0], tf.math.cumsum(i[1:] - i[:-1])), axis=0, name="rle_concat_positions")
    lengths = pos[1:] - pos[:-1]
    return pos[:-1], lengths


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.bool),
    tf.TensorSpec(shape=[], dtype=tf.int64)])
def invert_too_short_consecutive_false(mask, min_length):
    if min_length == 0:
        return mask
    pos, group_lengths = run_length_encoding(tf.cast(mask, tf.int32))
    true_or_too_short = tf.math.logical_or(tf.gather(mask, pos), group_lengths < min_length)
    new_mask = tf.repeat(true_or_too_short, group_lengths)
    tf.debugging.assert_equal(tf.size(mask), tf.size(new_mask))
    return new_mask


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)])
def framewise_rms_energy_vad_decisions(signal, sample_rate, frame_step_ms, min_non_speech_ms=0, strength=0.05, min_rms_threshold=1e-3, time_axis=0):
    """
    Compute energy based frame-wise VAD decisions by comparing the RMS value of each frame to the mean RMS of the whole signal, such that True means the frame is voiced and False unvoiced.
    VAD threshold is 'strength' multiplied by mean RMS, i.e. larger 'strength' values increase VAD aggressiveness.
    """
    # Partition signal into non-overlapping frames
    frame_step = ms_to_frames(sample_rate, frame_step_ms)
    frames = tf.signal.frame(signal, frame_step, frame_step, axis=time_axis)

    # Compute RMS and mean RMS over frames
    rms = root_mean_square(frames, axis=time_axis+1)
    mean_rms = tf.math.reduce_mean(rms, axis=time_axis, keepdims=True)

    # Compute VAD decisions using the RMS threshold
    threshold = strength * tf.math.maximum(min_rms_threshold, mean_rms)
    vad_decisions = rms > threshold

    # Check if there are too short sequences of positive VAD decisions and revert them to negative
    min_non_speech_frames = tf.cast(ms_to_frames(sample_rate, min_non_speech_ms) / frame_step, tf.int64)
    vad_decisions = invert_too_short_consecutive_false(vad_decisions, min_non_speech_frames)

    # Ensure the length VAD decisions is always equal to the number of signal frames/windows
    return tf.reshape(vad_decisions, [tf.shape(frames)[0]])


# # similar to kaldi mfcc vad but without a context window (for now):
# # https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/src/ivector/voice-activity-detection.cc
# # TODO something is wrong, this drops almost all frames
# @tf.function
# def framewise_mfcc_energy_vad_decisions(wav, spec_kwargs, melspec_kwargs, energy_threshold=5.0, energy_mean_scale=0.5):
#     tf.debugging.assert_rank(wav.audio, 1, message="Expected a single 1D signal (i.e. one mono audio tensor without explicit channels)")
#     S = spectrograms(Wav(tf.expand_dims(wav.audio, 0), tf.expand_dims(wav.sample_rate, 0)), **spec_kwargs)
#     S = melspectrograms(S, sample_rate=wav.sample_rate, **melspec_kwargs)
#     S = tf.math.log(S + 1e-6)
#     tf.debugging.assert_all_finite(S, "logmelspectrogram extraction failed, cannot compute mfcc energy vad")
#     mfcc = tf.signal.mfccs_from_log_mel_spectrograms(S)
#     log_energy = tf.squeeze(tf.signal.mfccs_from_log_mel_spectrograms(S)[..., 0])
#     tf.debugging.assert_rank(log_energy, 1, message="Failed to extract 0th MFCC coef")
#     mean_log_energy = tf.math.reduce_mean(log_energy)
#     vad_decisions = log_energy > (energy_threshold + energy_mean_scale * mean_log_energy)
#     return vad_decisions

# Cannot be a tf.function due to external Python object webrtcvad.Vad
def numpy_fn_get_webrtcvad_decisions(signal, sample_rate, pcm_data, vad_step, aggressiveness, min_non_speech_frames):
    assert 2 * signal.size == len(pcm_data), "signal length was {}, but pcm_data length was {}, when {} was expected (sample width 2)".format(signal.size, len(pcm_data), 2 * signal.size)
    vad_decisions = np.ones(signal.size // vad_step, dtype=np.bool)
    vad_step_bytes = 2 * vad_step
    vad = webrtcvad.Vad(aggressiveness)
    non_speech_begin = -1
    #TODO vectorize and remove loop with run_length_encoding
    for f, i in enumerate(range(0, len(pcm_data) - len(pcm_data) % vad_step_bytes, vad_step_bytes)):
        if not vad.is_speech(pcm_data[i:i+vad_step_bytes], sample_rate):
            vad_decisions[f] = False
            if non_speech_begin < 0:
                non_speech_begin = f
        else:
            if non_speech_begin >= 0 and f - non_speech_begin < min_non_speech_frames:
                # too short non-speech segment, revert all non-speech decisions up to f
                vad_decisions[np.arange(non_speech_begin, f)] = True
            non_speech_begin = -1
    return vad_decisions


def _count_wav_body_size(path_bytes):
    """
    https://github.com/mozilla/DeepSpeech/issues/2048#issuecomment-539518251
    """
    with wave.open(path_bytes.decode("utf-8"), 'r') as f_in:
        return f_in.getnframes() * f_in.getnchannels() * f_in.getsampwidth()


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
def wav_header_is_valid(path):
    """
    Return True if the file at 'path' is a valid wav file that can be decoded using 'read_wav' and False if it is not.
    It is assumed the wav file is less than 2 GiB.
    """
    file_contents = tf.io.read_file(path)
    if tf.strings.substr(file_contents, 0, 4) != "RIFF":
        return False
    else:
        wav_body_size = tf.cast(tf.numpy_function(_count_wav_body_size, [path], tf.int64), tf.int32)
        return wav_body_size + 44 == tf.strings.length(file_contents)
