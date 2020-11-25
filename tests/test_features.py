"""
Unit tests for lidbox.features.
"""
import pytest

import numpy as np
import tensorflow as tf

import lidbox.features as features


class TestFeatures(tf.test.TestCase):

    def test_feature_scaling(self):
        for rank in range(1, 5):
            for _ in range(300):
                delta = np.random.uniform(1, 1e3)
                min = np.random.uniform(-delta, delta)
                max = min + np.random.uniform(0, delta/2)
                x = np.random.normal(0, delta**2, size=np.random.randint(2, 20, size=rank))
                for axis in [None] + list(range(rank)):
                    y = features.feature_scaling(x, min, max, axis=axis).numpy()
                    assert not np.isnan(y).any()
                    assert y.shape == x.shape
                    assert np.abs(y.min(axis=axis) - min).max() < 1e-9
                    assert np.abs(y.max(axis=axis) - max).max() < 1e-9

    def test_cmvn(self):
        rank = 3
        for delta_magnitude in range(2, 7):
            for _ in range(200):
                delta = np.random.uniform(1, 10**delta_magnitude)
                x = np.random.uniform(-delta, delta, size=np.random.randint(1, 20, size=rank))
                for axis in range(rank):
                    y_m = features.cmvn(x, axis=axis, normalize_variance=False).numpy()
                    assert not np.isnan(y_m).any()
                    assert y_m.shape == x.shape
                    assert np.abs(y_m.mean(axis=axis)).max() < 1
                    y_mv = features.cmvn(x, axis=axis, normalize_variance=True).numpy()
                    assert not np.isnan(y_mv).any()
                    assert y_mv.shape == x.shape
                    assert np.abs(y_mv.mean(axis=axis)).max() < 0.1
                    assert y_mv.var(axis=axis).max() < 10

    def test_window_normalization(self):
        rank = 3
        for _ in range(50):
            delta = np.random.uniform(1, 1e3)
            x = np.random.uniform(-delta, delta, size=np.random.randint(1, 20, size=rank))
            axis = 1
            assert axis == 1, "TODO: test windows on arbitrary axis"
            for window_len in [-1] + list(range(2, x.shape[0]+1)):
                for normalize_variance in (True, False):
                    kwargs = dict(axis=axis, window_len=window_len, normalize_variance=normalize_variance)
                    y = features.window_normalization(x, **kwargs).numpy()
                    assert not np.isnan(y).any()
                    assert y.shape == x.shape
                    #TODO assert window means and variances
