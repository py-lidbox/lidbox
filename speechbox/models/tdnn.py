import tensorflow as tf

class TDNN(tf.keras.layers.Layer):
    """
    Fully connected layer that splices time-dependent input by subsampling the surrounding time context with a fixed set of indexes.
    Should correspond to the TDNN with subsampling defined by Peddinti et al. (2015) in http://www.danielpovey.com/files/2015_interspeech_aspire.pdf
    """
    def __init__(self, fc_units, subsample_offsets=[0], stride=1, name="tdnn", **dense_kwargs):
        super().__init__(name=name)
        tf.debugging.assert_rank(subsample_offsets, 1, message="subsample_offsets must be a 1D tensor/list of subsampling indexes, castable to tf.int32")
        self.fc = tf.keras.layers.Dense(fc_units, name=name + "_fc", **dense_kwargs)
        self.subsample_offsets = tf.sort(tf.cast(subsample_offsets, tf.int32))
        self.stride = tf.constant(stride, dtype=tf.int32)
        self._range_begin_offset = tf.constant(
            tf.math.abs(tf.math.minimum(0, tf.math.reduce_min(self.subsample_offsets))),
            dtype=tf.int32)
        self._range_end_offset = tf.constant(
            tf.math.maximum(0, tf.math.reduce_max(self.subsample_offsets)),
            dtype=tf.int32)

    def call(self, inputs):
        valid_input_indexes = tf.range(
            self._range_begin_offset,
            tf.shape(inputs)[1] - self._range_end_offset,
            self.stride)
        sample_indexes = self.subsample_offsets + tf.expand_dims(valid_input_indexes, -1)
        outputs = tf.gather(inputs, sample_indexes, axis=1)
        oshape = tf.shape(outputs)
        return self.fc(tf.reshape(outputs, (-1, oshape[-3], oshape[-2] * oshape[-1])))
