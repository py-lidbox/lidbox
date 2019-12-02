from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Metric, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
import tensorflow as tf
import numpy as np


# non-tf version
def avg_eer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return (0.5 * (fpr + fnr)).mean()

class OneHotAvgEER(Metric):
    """
    Average Equal Error Rate for one-hot encoded targets containing 'num_classes' of different classes.
    """
    def __init__(self, num_classes, name="avg_eer", **kwargs):
        super().__init__(name=name, **kwargs)
        # 4 metrics for each class
        self.class_metrics = [
            (TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives())
            for _ in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, **kwargs):
        for i, c in enumerate(self.class_metrics):
            for metric in c:
                metric.update_state(y_true[:,i], y_pred[:,i], **kwargs)

    def reset_states(self):
        for c in self.class_metrics:
            for metric in c:
                metric.reset_states()

    def result(self):
        @tf.function
        def eer(tp, tn, fp, fn):
            fp_rate = fp / (fp + tn)
            fn_rate = fn / (fn + tp)
            if tf.math.is_nan(fp_rate):
                fp_rate = tf.constant(1.0)
            if tf.math.is_nan(fn_rate):
                fn_rate = tf.constant(1.0)
            return 0.5 * (fp_rate + fn_rate)
        eer_by_label = [eer(*[m.result() for m in c]) for c in self.class_metrics]
        return tf.reduce_mean(eer_by_label)


class AverageDetectionCost(Metric):
    """
    TensorFlow implementation of C_avg equation 32 from
    Li, H., Ma, B. and Lee, K.A., 2013. Spoken language recognition: from fundamentals to practice. Proceedings of the IEEE, 101(5), pp.1136-1159.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6451097
    """
    def __init__(self, N, name="C_avg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.P_miss = [FalseNegatives() for _ in range(N)]
        self.P_fa = [[FalsePositives() for _ in range(N - 1)] for _ in range(N)]

    def update_state(self, y_true, y_pred, **kwargs):
        """
        Given a batch of true labels and predicted labels, update P_miss and P_fa for each label.
        """
        # Update false negatives for each target l
        for l, fn in enumerate(self.P_miss):
            fn.update_state(y_true[:,l], y_pred[:,l], **kwargs)
        # Update false positives for each target l and non-target m pair (l, m), such that l != m
        for l, p_fa in enumerate(self.P_fa):
            for m, fp in enumerate(p_fa):
                m += int(m >= l)
                fp.update_state(y_true[:,m], y_pred[:,m], **kwargs)

    def reset_states(self):
        for fn in self.P_miss:
            fn.reset_states()
        for fps in self.P_fa:
            for fp in fps:
                fp.reset_states()

    def result(self, C_miss=1.0, C_fa=1.0, P_tar=0.5):
        """
        Return final C_avg value as a scalar tensor.
        """
        avg_P_miss = tf.reduce_mean([fn.result() for fn in self.P_miss])
        avg_P_fa = tf.reduce_mean([tf.reduce_mean([fp.result() for fp in fps]) for fps in self.P_fa])
        return C_miss * P_tar * avg_P_miss + C_fa * (1 - P_tar) * avg_P_fa
