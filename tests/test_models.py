"""
Unit tests for lidbox.models.
"""
from hypothesis import given
from hypothesis.strategies import integers
from lidbox.testutil import spectrograms

from lidbox.models import (
    ap_lstm,
    bi_gru,
    clstm,
    cnn,
    convnet_extractor,
    crnn,
    dnn,
    lstm,
    multilevel_attention,
    spherespeaker,
    xvector,
    xvector_2d,
    xvector_extended,
    xvector_freq_attention,
)

import pytest
import numpy as np
import tensorflow as tf


def _assert_valid_model_output(module, x, num_outputs, **create_kw):
    m = module.create(x.shape[1:], num_outputs, **create_kw)
    for t in (False, True):
        y = m(x, training=t).numpy()
        assert not np.isnan(y).any(), "model output contained NaNs"
        assert y.shape == (x.shape[0], num_outputs), "model output has unexpected shape"


class TestModels(tf.test.TestCase):

    def setup_example(self):
        super().setUp()

    def teardown_example(self, ex):
        super().tearDown()

    @given(x=spectrograms(),
           num_lstm_units=integers(min_value=10, max_value=500))
    def test_ap_lstm(self, x, num_lstm_units):
        m = ap_lstm.create(x.shape[1:], num_lstm_units=num_lstm_units)
        for t in (False, True):
            y = m(x, training=t).numpy()
            assert not np.isnan(y).any()
            assert y.shape == (x.shape[0], 4*num_lstm_units)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_bi_gru(self, **kw):
        _assert_valid_model_output(bi_gru, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_clstm(self, **kw):
        _assert_valid_model_output(clstm, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_cnn(self, **kw):
        _assert_valid_model_output(cnn, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_convnet_extractor(self, **kw):
        _assert_valid_model_output(convnet_extractor, **kw)

    @given(x=spectrograms(min_shape=(1, 32, 32)),
           num_outputs=integers(min_value=1, max_value=100))
    def test_crnn(self, **kw):
        _assert_valid_model_output(crnn, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_dnn(self, **kw):
        _assert_valid_model_output(dnn, **kw)

    @given(x=spectrograms(max_shape=(64, 100, 128)),
           num_outputs=integers(min_value=1, max_value=100),
           num_units=integers(min_value=1, max_value=2000))
    def test_lstm(self, **kw):
        _assert_valid_model_output(lstm, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100),
           L=integers(min_value=2, max_value=20),
           H=integers(min_value=8, max_value=512))
    def test_multilevel_attention(self, **kw):
        _assert_valid_model_output(multilevel_attention, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100),
           embedding_dim=integers(min_value=2, max_value=2000))
    def test_spherespeaker(self, **kw):
        _assert_valid_model_output(spherespeaker, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_xvector(self, **kw):
        _assert_valid_model_output(xvector, **kw)

    @given(x=spectrograms(min_shape=(1, 1, 23)),
           num_outputs=integers(min_value=1, max_value=100))
    def test_xvector_2d(self, **kw):
        _assert_valid_model_output(xvector_2d, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_xvector_extended(self, **kw):
        _assert_valid_model_output(xvector_extended, **kw)

    @given(x=spectrograms(),
           num_outputs=integers(min_value=1, max_value=100))
    def test_xvector_freq_attention(self, **kw):
        _assert_valid_model_output(xvector_freq_attention, **kw)
