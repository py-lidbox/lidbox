import tensorflow as tf


class FalseNegativeRate(tf.keras.metrics.Metric):
    def __init__(self, thresholds, name="fnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives(thresholds=thresholds)
        self.fn = tf.keras.metrics.FalseNegatives(thresholds=thresholds)

    def reset_states(self):
        self.tp.reset_states()
        self.fn.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        self.tp.update_state(y_true, y_pred, **kwargs)
        self.fn.update_state(y_true, y_pred, **kwargs)

    def result(self):
        tp = self.tp.result()
        fn = self.fn.result()
        return tf.math.divide_no_nan(fn, tp + fn)


class FalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, thresholds, name="fpr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = tf.keras.metrics.TrueNegatives(thresholds=thresholds)
        self.fp = tf.keras.metrics.FalsePositives(thresholds=thresholds)

    def reset_states(self):
        self.tn.reset_states()
        self.fp.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        self.tn.update_state(y_true, y_pred, **kwargs)
        self.fp.update_state(y_true, y_pred, **kwargs)

    def result(self):
        tn = self.tn.result()
        fp = self.fp.result()
        return tf.math.divide_no_nan(fp, tn + fp)


class AveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, target_names, thresholds=None, name="avg_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precisions = [tf.keras.metrics.Precision(thresholds=thresholds, name="{}_precision".format(t)) for t in target_names]

    def reset_states(self):
        for p in self.precisions:
            p.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        for i, p in enumerate(self.precisions):
            p.update_state(y_true[:,i], y_pred[:,i], **kwargs)

    def result(self):
        return tf.math.reduce_mean([p.result() for p in self.precisions])

    def __iter__(self):
        for p in self.precisions:
            yield p


class AverageRecall(tf.keras.metrics.Metric):
    def __init__(self, target_names, thresholds=None, name="avg_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.recalls = [tf.keras.metrics.Recall(thresholds=thresholds, name="{}_recall".format(t)) for t in target_names]

    def reset_states(self):
        for r in self.recalls:
            r.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        for i, r in enumerate(self.recalls):
            r.update_state(y_true[:,i], y_pred[:,i], **kwargs)

    def result(self):
        return tf.math.reduce_mean([r.result() for r in self.recalls])

    def __iter__(self):
        for r in self.recalls:
            yield r


class AverageDetectionCost(tf.keras.metrics.Metric):
    """
    TensorFlow implementation of C_avg equation 32 from
    Li, H., Ma, B. and Lee, K.A., 2013. Spoken language recognition: from fundamentals to practice. Proceedings of the IEEE, 101(5), pp.1136-1159.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6451097
    """
    def __init__(self, target_names, theta_det=(0.5,), C_miss=1.0, C_fa=1.0, P_tar=0.5, name="C_avg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.P_miss = [FalseNegativeRate(theta_det, name="{}_fnr".format(t)) for t in target_names]
        self.P_fa = [[FalsePositiveRate(theta_det, name="{}_as_{}_fpr".format(l, m)) for m in target_names if m != l] for l in target_names]
        self.C_miss = C_miss
        self.C_fa = C_fa
        self.P_tar = P_tar

    def reset_states(self):
        for fnr in self.P_miss:
            fnr.reset_states()
        for p_fa in self.P_fa:
            for fpr in p_fa:
                fpr.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        """
        Given a batch of true scores and predicted scores, update P_miss and P_fa for each score.
        """
        # Update false negatives for each target l
        for l, fnr in enumerate(self.P_miss):
            fnr.update_state(y_true[:,l], y_pred[:,l], **kwargs)
        # Update false positives for each target l and non-target m pair (l, m), such that l != m
        for l, p_fa in enumerate(self.P_fa):
            for m, fpr in enumerate(p_fa):
                m += int(m >= l)
                fpr.update_state(y_true[:,m], y_pred[:,m], **kwargs)

    def result(self):
        """
        Return smallest C_avg value using all given thresholds.
        To use different C_avg parameter values, set e.g. cavg.P_tar = 0.4 before calling cavg.result().
        """
        # FNR for each target, evaluated with all given thresholds
        P_miss = [fnr.result() for fnr in self.P_miss]
        # FPR for each target and non-target pair, evaluated with all given thresholds
        P_fa = [tf.reduce_mean([fpr.result() for fpr in p_fa], axis=0) for p_fa in self.P_fa]
        # Reduce mean over all targets
        avg_P_miss = tf.reduce_mean(P_miss, axis=0)
        avg_P_fa = tf.reduce_mean(P_fa, axis=0)
        C_avgs = self.C_miss * self.P_tar * avg_P_miss + self.C_fa * (1 - self.P_tar) * avg_P_fa
        # Return minimum
        self.min_index = tf.argmin(C_avgs)
        return C_avgs[self.min_index]


class EqualErrorRate(tf.keras.metrics.Metric):
    def __init__(self, thresholds=None, name="eer", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fnr = FalseNegativeRate(thresholds)
        self.fpr = FalsePositiveRate(thresholds)

    def reset_states(self):
        self.fnr.reset_states()
        self.fpr.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        self.fnr.update_state(y_true, y_pred, **kwargs)
        self.fpr.update_state(y_true, y_pred, **kwargs)

    def result(self):
        # Index of the threshold where the absolute difference of the FNR and FPR values is smallest
        # If a single threshold is being used, the fnr and fpr results will be scalars, so we add a singleton dim
        fnr = tf.expand_dims(self.fnr.result(), -1)
        fpr = tf.expand_dims(self.fpr.result(), -1)
        self.min_index = tf.squeeze(tf.argmin(tf.math.abs(fnr - fpr)))
        # Return the mean of FPR and FPR at the min index
        return 0.5 * (fnr[self.min_index] + fpr[self.min_index])


class AverageEqualErrorRate(tf.keras.metrics.Metric):
    def __init__(self, target_names, thresholds=None, name="avg_eer", **kwargs):
        super().__init__(name=name, **kwargs)
        # EERs for each target
        self.eers = [EqualErrorRate(thresholds=thresholds, name="{}_eer".format(t)) for t in target_names]

    def reset_states(self):
        for eer in self.eers:
            eer.reset_states()

    def update_state(self, y_true, y_pred, **kwargs):
        for l, eer in enumerate(self.eers):
            eer.update_state(y_true[:,l], y_pred[:,l], **kwargs)

    def result(self):
        return tf.reduce_mean([eer.result() for eer in self.eers])

    def __iter__(self):
        for eer in self.eers:
            yield eer
