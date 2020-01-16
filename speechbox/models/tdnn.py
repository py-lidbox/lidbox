from tensorflow.keras.layers import (
    Dense,
    Layer,
)
import tensorflow as tf


class TDNN(Layer):
    """
    Fully connected layer that splices time-dependent input by subsampling the surrounding time context with a fixed set of indices.
    Should correspond to the TDNN defined by Waibel et al. (1989) in https://www.cs.toronto.edu/~hinton/absps/waibelTDNN.pdf, with the subsampling scheme used by Peddinti et al. (2015) in http://www.danielpovey.com/files/2015_interspeech_aspire.pdf
    """
    def __init__(self, fc_units, subsample_offsets=[0], stride=1, name="tdnn", padding="valid", **dense_kwargs):
        super().__init__(name=name)
        tf.debugging.assert_rank(subsample_offsets, 1, message="subsample_offsets must be a 1D tensor/list of subsampling indices, castable to tf.int32")
        assert padding in ("valid",), "unknown padding type '{}'".format(padding)
        self.padding = padding
        self.subsample_offsets = tf.cast(subsample_offsets, tf.int32)
        self.stride = tf.constant(stride, dtype=tf.int32)
        self.output_fc = Dense(fc_units, **dense_kwargs)

    # TODO implement edge padding
    # TODO update with partial edge handling when tf.gather supports non-zero axes for ragged input
    def _get_padded_window_indices(self, input_shape):
        input_indices = tf.range(0, input_shape[1], self.stride)
        sample_indices = self.subsample_offsets + tf.expand_dims(input_indices, -1)
        if self.padding == "same":
            pass
        elif self.padding == "causal":
            pass
        else:
            valid_windows = tf.reduce_all(tf.logical_and(sample_indices >= 0, sample_indices < input_shape[1]), axis=1)
            return tf.boolean_mask(sample_indices, valid_windows)

    def build(self, input_shape):
        print("call", self.build, input_shape)
        super().build(input_shape)
        sample_shape = tf.shape(self._get_padded_window_indices(input_shape))
        self.output_fc.build(tf.concat(([input_shape[0]], sample_shape), axis=0))

    def call(self, inputs):
        print("call", self.call, inputs)
        sample_indices = self._get_padded_window_indices(tf.shape(inputs))
        outputs = tf.gather(inputs, sample_indices, axis=1)
        oshape = tf.shape(outputs)
        outputs = tf.reshape(outputs, (-1, oshape[1], oshape[2] * oshape[3]))
        return self.output_fc(outputs)
