"""
Unit tests for lidbox.features.audio.
"""
import os
import tempfile

import pytest

import librosa
import numpy as np
import scipy.signal
import tensorflow as tf

from lidbox.features import audio


audiofiles = [
    "noisy_100hz_sine.wav",
    "noisy_200hz_sine.wav",
    "noisy_300hz_sine.wav",
    "noisy_400hz_sine.wav",
    "noise.wav",
]
audiofiles = [os.path.join("tests", "audio", f) for f in audiofiles]


class TestFeaturesAudio(tf.test.TestCase):

    def test_read_wav(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            assert not np.isnan(s.numpy()).any(), "NaNs in signal"
            assert s.shape == (3*16000,), "unexpected signal shape"
            assert r == 16000, "unexpected sample rate"

    def test_read_mp3(self):
        for wavpath in audiofiles:
            mp3path = wavpath.rsplit(".wav", 1)[0] + ".mp3"
            s, r = audio.read_mp3(mp3path)
            assert not np.isnan(s.numpy()).any(), "NaNs in signal"
            assert s.shape == (49536,), "unexpected signal shape"
            assert r == 16000, "unexpected sample rate"

    def test_resample(self):
        for path in audiofiles:
            s1, r1 = audio.read_wav(path)
            s2, r2 = audio.pyfunc_resample(s1, r1, 2*r1)
            assert r2 == 2*r1
            assert not np.isnan(s2.numpy()).any(), "NaNs after resampling"
            assert len(s2.shape) == len(s1.shape), "signal shape changed after resampling"
            assert s2.shape[0] == 2*s1.shape[0], "unexpected signal length after resampling"

    def test_dBFS_to_linear(self):
        for i, level in enumerate(range(0, 200, 20)):
            a = audio.dBFS_to_linear(level)
            assert not np.isnan(a.numpy()).any()
            assert np.abs(a - 10 ** i) < 1e-6

    def test_peak_normalize(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            s1 = s + np.random.normal(0, 10, s.shape)
            for level in range(0, -10, -1):
                s2 = audio.peak_normalize(s1, dBFS=level)
                assert not np.isnan(s2.numpy()).any()
                assert np.max(np.abs(s2)) <= audio.dBFS_to_linear(level), "maximum amplitude cannot exceed given dBFS level after peak normalization"

    @pytest.mark.skip(reason="TODO: random seeds")
    def test_random_gaussian_fir_filter(self):
        pass

    def test_write_mono_wav(self):
        for inpath in audiofiles:
            s, r = audio.read_wav(inpath)
            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = os.path.join(tmpdir, os.path.basename(inpath))
                wrotepath = audio.write_mono_wav(outpath, s, r)
                assert os.path.exists(outpath)
                assert wrotepath == outpath
                assert librosa.get_duration(filename=outpath, sr=None) == (s.shape[0] / r)
                assert librosa.get_samplerate(outpath) == r
                s1, r1 = librosa.load(outpath, sr=None)
                assert not np.isnan(s1).any()
                assert s1.shape == s.shape
                assert r1 == r

    def test_wav_to_pcm_data(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            h, b = audio.wav_to_pcm_data(s, r)
            assert len(h.numpy()) == 44, "unexpected wav header length"
            assert h.numpy()[:4].decode("ascii") == "RIFF", "wav header did not begin with 'RIFF'"
            assert len(b.numpy()) == 2 * s.shape[0], "unexpected wav data length, expected sample width of 2"

    @pytest.mark.skip(reason="TODO")
    def test_snr_mixer(self):
        pass

    def test_fft_frequencies(self):
        for sr in range(4000, 60000, 4000):
            for n_fft in (2**i for i in range(1, 13)):
                a = audio.fft_frequencies(sr, n_fft)
                b = librosa.fft_frequencies(sr, n_fft)
                assert np.abs(a - b).max() < 1e-9

    def test_log10(self):
        for rank in range(1, 5):
            for _ in range(5):
                x = np.random.normal(1e6, 1e4, size=np.random.randint(1, 10, size=rank))
                x = np.maximum(1e-12, x)
                y1 = np.log10(x)
                y2 = audio.log10(tf.constant(x, tf.float32))
                assert np.abs(y1 - y2.numpy()).max() < 1e-6

    def test_power_to_db(self):
        for top_db in range(10, 110, 10):
            for path in audiofiles:
                s, r = audio.read_wav(path)
                _, _, stft = scipy.signal.stft(s)
                powspec = np.abs(stft)**2
                dbspec = audio.power_to_db(np.expand_dims(powspec, 0), top_db=float(top_db))[0].numpy()
                assert not np.isnan(dbspec).any()
                assert dbspec.max() <= 0

    def test_ms_to_frames(self):
        for sr in range(1000, 60000, 1000):
            for ms in range(1, 5000, 100):
                nframes = (sr // 1000) * ms
                assert audio.ms_to_frames(sr, ms).numpy() == nframes

    def test_spectrograms(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            for len_ms in range(20, 101, 20):
                for n_fft in (256, 512, 1024, 2048):
                    if n_fft < audio.ms_to_frames(r, len_ms):
                        continue
                    step_ms = len_ms // 2
                    powspec = audio.spectrograms(np.expand_dims(s, 0), r,
                            frame_length_ms=len_ms,
                            frame_step_ms=step_ms,
                            fft_length=n_fft)[0]
                    assert not np.isnan(powspec.numpy()).any()
                    assert powspec.shape[0] == s.shape[0] // audio.ms_to_frames(r, step_ms) - 1
                    assert powspec.shape[1] == n_fft // 2 + 1

    def test_linear_to_mel(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            for num_mel_bins in range(10, 100, 15):
                powspecs = audio.spectrograms(np.expand_dims(s, 0), r)
                melspec = audio.linear_to_mel(powspecs, r, num_mel_bins=num_mel_bins)[0]
                assert not np.isnan(melspec.numpy()).any()
                assert melspec.shape[0] == powspecs[0].shape[0]
                assert melspec.shape[1] == num_mel_bins

    def test_root_mean_square(self):
        for _ in range(100):
            x = np.random.normal(0, 5, size=np.random.randint(1, 10, size=2))
            rms1 = np.sqrt(np.mean(np.square(np.abs(x)), axis=-1))
            rms2 = audio.root_mean_square(x, axis=-1).numpy()
            assert not np.isnan(rms2).any()
            assert np.abs(rms1 - rms2).max() < 1e-5

    #TODO generalize with random data
    def test_run_length_encoding(self):
        pos, length = audio.run_length_encoding(np.array([1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 7]))
        assert (pos.numpy() == np.array([0, 3, 6, 7, 8, 9, 11])).all()
        assert (length.numpy() == np.array([3, 3, 1, 1, 1, 2, 1])).all()

    @pytest.mark.skip(reason="TODO")
    def test_invert_too_short_consecutive_false(self):
        pass

    def test_framewise_rms_energy_vad_decisions(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            vad = audio.framewise_rms_energy_vad_decisions(s, r, 25)
            assert (vad.numpy() == 1).all()
        vad = audio.framewise_rms_energy_vad_decisions(np.zeros(3*16000), 16000, 25)
        assert (vad.numpy() == 0).all()

    def test_remove_silence(self):
        for path in audiofiles:
            s, r = audio.read_wav(path)
            s1 = audio.remove_silence(s, r)
            assert not np.isnan(s1.numpy()).any()
            assert s1.shape == s.shape
        s1 = audio.remove_silence(np.zeros(3*16000), 16000)
        assert not np.isnan(s1.numpy()).any()
        assert tf.size(s1) == 0

    @pytest.mark.skip(reason="TODO")
    def test_numpy_fn_get_webrtcvad_decisions(self):
        pass
