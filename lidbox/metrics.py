import tensorflow as tf

DEBUG = False


class AverageDetectionCost(tf.keras.metrics.Metric):
    """
    TensorFlow implementation of C_avg equation 32 from
    Li, H., Ma, B. and Lee, K.A., 2013. Spoken language recognition: from fundamentals to practice. Proceedings of the IEEE, 101(5), pp.1136-1159.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6451097

    This implementation is not limited to language identification, any float one-hot encoded labels should work.

    Args:
        N: Amount of labels
        thresholds: Array of 1 or more decision thresholds (theta_det). Make sure the thresholds match the output scores (e.g. log-likelihoods) of the model.
    """
    def __init__(self, N, thresholds, C_miss=1.0, C_fa=1.0, P_tar=0.5, name="C_avg", **kwargs):
        super().__init__(name=name, **kwargs)
        tf.debugging.assert_greater_equal(N, 2, message="C_avg is undefined for less than 2 classes.")
        tf.debugging.assert_rank(thresholds, 1, message="Thresholds must be an array of decision scores.")
        num_thresholds = len(thresholds)
        # All positives for N labels
        self.fn = self.add_weight(
                name="fn",
                shape=[N, num_thresholds],
                initializer="zeros")
        self.tp = self.add_weight(
                name="tp",
                shape=[N, num_thresholds],
                initializer="zeros")
        # All negatives for N * N label pairs,
        # the l == m case will always be zero and is asserted when the result is requested
        self.fp_pairs = self.add_weight(
                name="fp_pairs",
                shape=[N, N, num_thresholds],
                initializer="zeros")
        self.tn_pairs = self.add_weight(
                name="tn_pairs",
                shape=[N, N, num_thresholds],
                initializer="zeros")
        self.thresholds = tf.constant(thresholds, dtype=tf.float32, name="thresholds")
        self.C_miss = C_miss
        self.C_fa = C_fa
        self.P_tar = P_tar

    def reset_states(self):
        for var in ("fn", "tp", "fp_pairs", "tn_pairs"):
            getattr(self, var).assign(tf.zeros_like(getattr(self, var)))

    def update_state(self, true_positives, predictions, **kwargs):
        """
        Update P_miss and P_fa counters for a given batch of true labels and predicted scores.
        """
        # Save the indices of correct labels for scattering updates into P_fa pair counters
        label_indices = tf.expand_dims(tf.math.argmax(true_positives, axis=-1), -1)
        true_positives = tf.expand_dims(true_positives, -1)
        true_negatives = tf.cast(~tf.cast(true_positives, tf.bool), tf.float32)
        predictions = tf.expand_dims(tf.cast(predictions, tf.float32), -1)
        pred_positives = tf.cast(predictions >= self.thresholds, tf.float32)
        pred_negatives = tf.cast(predictions < self.thresholds, tf.float32)
        # Update false negative counters for P_miss
        tp = pred_positives * true_positives
        fn = pred_negatives * true_positives
        self.tp.assign_add(tf.math.reduce_sum(tp, axis=0))
        self.fn.assign_add(tf.math.reduce_sum(fn, axis=0))
        # Update false positive counters for P_fa
        fp = pred_positives * true_negatives
        tn = pred_negatives * true_negatives
        self.fp_pairs.scatter_nd_add(label_indices, fp)
        self.tn_pairs.scatter_nd_add(label_indices, tn)

    def result(self):
        """
        Return smallest C_avg value using all given thresholds.
        """
        if DEBUG:
            self._assert_P_fa()
        # Average false negative rate over all labels for all given thresholds
        P_miss = tf.math.reduce_mean(
                tf.math.divide_no_nan(
                    self.fn,
                    self.fn + self.tp),
                axis=0)
        # Average false positive rates over all label pairs, then for all labels, for all given thresholds
        # The l == m case will always be zeros and is ignored when computing the average over all label pairs
        N_minus_1 = tf.cast(tf.shape(self.fp_pairs)[0] - 1, tf.float32)
        P_fa = tf.math.reduce_mean(
                tf.math.divide_no_nan(
                    tf.math.reduce_sum(
                        tf.math.divide_no_nan(
                            self.fp_pairs,
                            self.fp_pairs + self.tn_pairs),
                        axis=1),
                    N_minus_1),
                axis=0)
        # Average detection cost for all given thresholds
        C_avg = self.C_miss * self.P_tar * P_miss + self.C_fa * (1 - self.P_tar) * P_fa
        if DEBUG:
            tf.print("P_miss", P_miss, summarize=-1)
            tf.print("P_fa", P_fa, summarize=-1)
            tf.print("C_avg", C_avg, summarize=-1)
        return tf.math.reduce_min(C_avg)

    def _assert_P_fa(self):
        tf.print(self.__class__.__name__, "asserting that l == m pairs are not included in P_fa")
        # All l == m pairs should always be zeros (we want to have N*(N-1) results)
        indices = tf.tile(tf.expand_dims(tf.range(0, tf.shape(self.fp_pairs)[0]), -1), [1, 2])
        tf.debugging.assert_equal(tf.math.reduce_sum(tf.gather_nd(self.fp_pairs, indices)), 0.0, message="Failed to compute false positive pairs")
        tf.debugging.assert_equal(tf.math.reduce_sum(tf.gather_nd(self.tn_pairs, indices)), 0.0, message="Failed to compute true negative pairs")
        tf.print(self.__class__.__name__, "P_fa ok")


class SparseAverageDetectionCost(AverageDetectionCost):

    def update_state(self, true_positives, predictions, **kwargs):
        N = tf.shape(self.fn)[0]
        true_positives_dense = tf.one_hot(tf.cast(true_positives, tf.int32), N)
        super().update_state(true_positives_dense, predictions, **kwargs)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    true_positives = tf.constant([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        ], tf.float32)
    predictions = tf.math.log(tf.constant([
        [.1, .2, .9],
        [.9, .2, .0],
        [.1, .9, .0],
        [.2, .8, .5],
        [.6, .3, .1],
        [.1, .0, .7],
        [.1, .0, .7],
        [.9, .1, .0],
        ], tf.float32))
    from time import perf_counter
    begin = perf_counter()
    cavg = AverageDetectionCost(3, [tf.math.log(x).numpy() for x in [0.05, 0.4, 0.6, 0.95]])
    cavg.update_state(true_positives, predictions)
    res = cavg.result().numpy()
    end = perf_counter() - begin
    print("min cavg: {}, took {:.6f} sec".format(res, end))
    cavg.reset_states()
    assert cavg.result().numpy() == 0.0
