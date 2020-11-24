"""
Unit tests for lidbox.models.
"""
from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, composite, lists
from hypothesis.extra.numpy import arrays as np_arrays

import numpy as np
import tensorflow as tf

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
    xvector_2d_skip,
    xvector_2d_v2,
    xvector_extended,
    xvector_freq_attention,
    xvector_mobilenet,
    xvector_resnet,
    xvector_tfjs,
)


@composite
def model_input(draw):
    shape = draw(lists(integers(min_value=1, max_value=100), min_size=3, max_size=3))
    return draw(np_arrays(np.float32, shape, elements=dict(min_value=-1e3, max_value=1e3)))


class TestModelsValidOutput:

    @given(x=model_input(), num_lstm_units=integers(min_value=10, max_value=500))
    @settings(max_examples=20, deadline=timedelta(milliseconds=2000))
    def test_ap_lstm(self, x, num_lstm_units):
        m = ap_lstm.create(x.shape[1:], num_lstm_units=num_lstm_units)
        for t in (False, True):
            y = m(x, training=t).numpy()
            assert not np.isnan(y).any()
            assert y.shape == (x.shape[0], 4*num_lstm_units)

    # @given(x=model_input(), num_outputs=integers(min_value=1, max_value=1000))
    # @settings(max_examples=20, deadline=timedelta(milliseconds=2000))
    # def test_bi_gru(self, x, num_outputs):
    #     m = bi_gru.create(x.shape[1:], num_outputs)
    #     for t in (False, True):
    #         y = m(x, training=t).numpy()
    #         assert not np.isnan(y).any()
    #         assert y.shape == (x.shape[0], num_outputs)

    #TODO generalize
    @given(x=model_input(), num_outputs=integers(min_value=1, max_value=1000))
    @settings(max_examples=20, deadline=timedelta(milliseconds=2000))
    def test_clstm(self, x, num_outputs):
        m = clstm.create(x.shape[1:], num_outputs)
        for t in (False, True):
            y = m(x, training=t).numpy()
            assert not np.isnan(y).any()
            assert y.shape == (x.shape[0], num_outputs)

    @given(x=model_input(), num_outputs=integers(min_value=1, max_value=1000))
    @settings(max_examples=20, deadline=timedelta(milliseconds=2000))
    def test_cnn(self, x, num_outputs):
        m = cnn.create(x.shape[1:], num_outputs)
        for t in (False, True):
            y = m(x, training=t).numpy()
            assert not np.isnan(y).any()
            assert y.shape == (x.shape[0], num_outputs)

    @given(x=model_input(), num_outputs=integers(min_value=1, max_value=1000))
    @settings(max_examples=20, deadline=timedelta(milliseconds=2000))
    def test_convnet_extractor(self, x, num_outputs):
        m = convnet_extractor.create(x.shape[1:], num_outputs)
        for t in (False, True):
            y = m(x, training=t).numpy()
            assert not np.isnan(y).any()
            assert y.shape == (x.shape[0], num_outputs)
