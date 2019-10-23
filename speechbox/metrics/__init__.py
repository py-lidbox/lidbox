from tensorflow.keras.metrics import Metric, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow import math as tf_math


class EqualErrorRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics = (
            TruePositives(),
            TrueNegatives(),
            FalsePositives(),
            FalseNegatives()
        )

    def __iter__(self):
        for m in self._metrics:
            yield m

    def update_state(self, y_true, y_pred, **kwargs):
        for m in self:
            m.update_state(y_true, y_pred, **kwargs)

    def reset_states(self):
        for m in self:
            m.reset_states()

    def result(self):
        tp, tn, fp, fn = [m.result() for m in self]
        fp_rate = tf_math.divide_no_nan(fp, fp + tn)
        fn_rate = tf_math.divide_no_nan(fn, fn + tp)
        return 0.5 * (fp_rate + fn_rate)


class AverageDetectionCost(Metric):
    """
    C_avg as defined in equation 32 in
    Li, H., Ma, B. and Lee, K.A., 2013. Spoken language recognition: from fundamentals to practice. Proceedings of the IEEE, 101(5), pp.1136-1159.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6451097

    The following variables have been renamed:
    num_targets == N
    P_target == P_tar
    """
    def __init__(self, num_targets, C_miss=1.0, C_fa=1.0, P_target=0.5, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError
        # P_miss = fn
        # P_fa = fp

    def update_state(self, y_true, y_pred, **kwargs):
        #FIXME just pseudocode
        # for L, (yt_batch, yp_batch) in enumerate(zip(y_true_batches, y_pred_batches)):
            # P_miss[L] = false_negative_rate(yt_batch, yp_batch)
        pass

    def reset_states(self):
        pass

    def result(self):
        return 0
