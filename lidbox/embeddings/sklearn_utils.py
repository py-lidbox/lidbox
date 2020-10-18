import collections
import logging
import os

from mpl_toolkits.mplot3d import Axes3D
from plda import Classifier as PLDAClassifier
import colorcet
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis
import sklearn.naive_bayes
import sklearn.preprocessing
import tensorflow as tf

import lidbox.metrics


logger = logging.getLogger(__name__)

categorical_cmap = colorcet.glasbey_category10


class PLDA(PLDAClassifier):
    def transform(self, X):
        return self.model.transform(X, from_space='D', to_space='U_model')
    def __str__(self):
        return "PLDA: {:d} -> {:d} -> {:d} -> {:d} (PCA preprocessing with {} coefs)".format(
                *[self.model.get_dimensionality(space) for space in ("D", "X", "U", "U_model")],
                self.model.pca.n_components if self.model.pca else None)


def pca_scatterplot_by_label(label2sample, pca):
    plt.rcParams["font.size"] = 28
    assert pca.n_components in (2, 3), "PCA plot with n_components = %d not implemented, must be 2 or 3".format(pca.n_components)
    scatter_kw = dict(s=100, alpha=0.7)
    if pca.n_components == 2:
        fig, ax = plt.subplots(figsize=(20, 20))
        for (label, vecs), color in zip(label2sample.items(), categorical_cmap):
            vecs = pca.transform(vecs)
            ax.scatter(vecs[:,0], vecs[:,1], c=[color], label=label, edgecolors='none', **scatter_kw)
            ax.set_title("Embeddings in PLDA model space, projected with 2-dim PCA")
        # ax.set_frame_on(False)
    else:
        fig = plt.figure(figsize=(20, 20))
        ax = Axes3D(fig)
        for (label, vecs), color in zip(label2sample.items(), categorical_cmap):
            vecs = pca.transform(vecs)
            ax.scatter3D(vecs[:,0], vecs[:,1], zs=vecs[:,2], c=[color], label=label, **scatter_kw)
        ax.text2D(0.5, 1.0, "Embeddings in PLDA model space, projected with 3-dim PCA", transform=ax.transAxes)
    ax.legend()
    return fig


def plot_embedding_demo(data, target2label, label2sample, pca=None, output_figure_dir=None):
    plt.rcParams["font.size"] = 28
    def _write_and_close(fig, name):
        path = os.path.join(output_figure_dir, name)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close('all')
        logger.info("Wrote embedding demo to '%s'", path)
    labels = list(label2sample.keys())
    assert len(labels) <= len(categorical_cmap), "too many labels ({}) for colormap {} ({})".format(len(labels), repr(categorical_cmap), len(categorical_cmap))
    pixel_scaler = mcolors.Normalize(data["X"].min(), data["X"].max())
    if sum(vecs.shape[0] for vecs in label2sample.values()) > sum(vecs.shape[1] for vecs in label2sample.values()):
        subplot_kw = dict(nrows=1, ncols=len(labels), sharex=False, sharey=True)
    else:
        subplot_kw = dict(nrows=len(labels), ncols=1, sharex=True, sharey=False)
    fig, axes = plt.subplots(figsize=(20, 20), gridspec_kw=dict(hspace=0.01, wspace=0.01), **subplot_kw)
    for (label, vecs), ax in zip(label2sample.items(), axes):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(label)
        ax.set_frame_on(False)
        im = ax.imshow(vecs, cmap="RdBu_r", norm=pixel_scaler)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    cbar = fig.colorbar(
            im,
            cax=fig.add_axes([0.83, 0.1, 0.02, 0.8]),
            ticks=[pixel_scaler.vmin, 0, pixel_scaler.vmax])
    cbar.outline.set_visible(False)
    if pca:
        if "2D" in pca:
            pca_2d_plot = pca_scatterplot_by_label(label2sample, pca["2D"])
        if "3D" in pca:
            pca_3d_plot = pca_scatterplot_by_label(label2sample, pca["3D"])
    if output_figure_dir is not None:
        os.makedirs(output_figure_dir, exist_ok=True)
        _write_and_close(fig, "embeddings-PLDA-model-space.png")
        if pca:
            if "2D" in pca:
                _write_and_close(pca_2d_plot, "embeddings-PCA-2D.png")
            if "3D" in pca:
                _write_and_close(pca_3d_plot, "embeddings-PCA-3D.png")


def get_lda_scores(lda, test):
    if isinstance(lda, PLDA):
        pred, log_pred = lda.predict(test["X"])
    else:
        pred, log_pred = lda.predict(test["X"]), lda.predict_log_proba(test["X"])
    cce = tf.keras.losses.sparse_categorical_crossentropy(test["y"], log_pred, from_logits=True)
    cce = tf.math.reduce_mean(cce)
    accuracy = (pred == test["y"]).mean()
    return accuracy, cce.numpy()


def fit_lda(train, test):
    logger.info("Fitting LDA to train_X %s train_y %s", train["X"].shape, train["y"].shape)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(train["X"], train["y"])
    logger.info(
            "Done: %s\n  accuracy %.3f\n  categorical crossentropy %.3f",
            lda,
            *get_lda_scores(lda, test))
    return lda


def fit_plda(train, test, n_components=None):
    logger.info("Fitting PLDA to train_X %s train_y %s, using "
                + ("{} PCA components".format(n_components) if n_components else "as many PCA components as possible")
                + " for preprocessing.",
                train["X"].shape,
                train["y"].shape)
    plda = PLDA()
    plda.fit_model(train["X"], train["y"], n_principal_components=n_components)
    logger.info(
            "Done: %s\n  accuracy %.3f\n  categorical crossentropy %.3f",
            plda,
            *get_lda_scores(plda, test))
    return plda


def fit_plda_gridsearch(train, test, grid):
    logger.info("Performing grid search over %d different principal components for PLDA: %s", len(grid), ', '.join(str(n) for n in grid))
    best_plda, best_loss = None, float("inf")
    for n in grid:
        plda = fit_plda(train, test, n_components=n)
        _, cce = get_lda_scores(plda, test)
        if cce < best_loss:
            logger.info("New best at categorical crossentropy %.3f with:\n  %s", cce, plda)
            best_plda, best_loss = plda, cce
    return best_plda


def reduce_dimensions(train, test, dim_reducer):
    logger.info("Reducing train_X %s and test_X %s dimensions with:\n  %s",
            train["X"].shape,
            test["X"].shape,
            dim_reducer)
    train["X"] = dim_reducer.transform(train["X"])
    test["X"] = dim_reducer.transform(test["X"])
    logger.info("After dimension reduction: train_X %s, test_X %s", train["X"].shape, test["X"].shape)


def draw_random_sample(train, test, labels, target2label, sample_size=100):
    logger.info("Choosing %d random demo utterances per label for %d labels from train_X %s and test_X %s",
            sample_size,
            len(labels),
            train["X"].shape,
            test["X"].shape)
    label2sample = {}
    for split, data in (("train", train), ("test", test)):
        label2vecs = collections.defaultdict(list)
        for x, y in zip(data["X"], data["y"]):
            label2vecs[target2label[y]].append(x)
        label2vecs = {l: np.stack(vecs) for l, vecs in label2vecs.items()}
        label2vecs = {l: vecs[np.random.choice(np.arange(0, vecs.shape[0]), size=sample_size, replace=False)] for l, vecs in label2vecs.items()}
        label2sample[split] = collections.OrderedDict((l, label2vecs[l]) for l in sorted(labels) if l in label2vecs)
    return label2sample


def fit_classifier(train, test, labels, config, target2label, Classifier, n_plda_coefs=None, plot_demo=True):
    scaler = sklearn.preprocessing.StandardScaler()
    logger.info("Fitting scaler to train_X %s:\n  %s", train["X"].shape, scaler)
    scaler.fit(train["X"])
    train["X"] = scaler.transform(train["X"])
    test["X"] = scaler.transform(test["X"])

    dim_reducer = fit_plda(train, test, n_components=n_plda_coefs)
    logger.info("Reducing train_X %s and test_X %s dimensions with:\n  %s",
            train["X"].shape,
            test["X"].shape,
            dim_reducer)
    train["X"] = dim_reducer.transform(train["X"])
    test["X"] = dim_reducer.transform(test["X"])

    logger.info("Normalizing train_X %s, test_X %s", train["X"].shape, test["X"].shape)
    train["X"] = sklearn.preprocessing.normalize(train["X"])
    test["X"] = sklearn.preprocessing.normalize(test["X"])

    if plot_demo:
        logger.info("Drawing random sample of embeddings and plotting demo figures")
        pca = {
                "2D": sklearn.decomposition.PCA(n_components=2, whiten=False),
                "3D": sklearn.decomposition.PCA(n_components=3, whiten=False),
        }
        for p in pca.values():
            logger.info("Fitting PCA to train_X %s:\n  %s", train["X"].shape, p)
            p.fit(train["X"])
        label2sample = draw_random_sample(train, test, labels, target2label)
        demo_dir = os.path.join(
                config["sklearn_experiment"]["cache_directory"],
                config["sklearn_experiment"]["model"]["key"],
                config["sklearn_experiment"]["name"],
                "figures")
        plot_embedding_demo(train, target2label, label2sample["train"], pca, os.path.join(demo_dir, "train"))
        plot_embedding_demo(test, target2label, label2sample["test"], pca, os.path.join(demo_dir, "test"))

    classifier = Classifier()
    logger.info("Fitting classifier to train_X %s and train_y %s:\n  %s",
         train["X"].shape,
         train["y"].shape,
         classifier)
    classifier.fit(train["X"], train["y"])

    return {
        "scaler": scaler,
        "dim_reducer": dim_reducer,
        "classifier": classifier
    }


def predict_with_trained_classifier(unlabeled, config, target2label, pipeline):
    X = unlabeled["X"]
    if "scaler" in pipeline:
        logger.info("Scaling input %s with %s", X.shape, pipeline["scaler"])
        X = pipeline["scaler"].transform(X)
    if "dim_reducer" in pipeline:
        logger.info("Reducing dimensions of %s with %s", X.shape, pipeline["dim_reducer"])
        X = pipeline["dim_reducer"].transform(X)
    logger.info("Normalizing %s", X.shape)
    X = sklearn.preprocessing.normalize(X)

    logger.info("Predicting log probabilties for %s with %s", X.shape, pipeline["classifier"])
    predictions = pipeline["classifier"].predict_log_proba(X)
    predictions = np.maximum(predictions, -100)
    return predictions


def joblib_dir_from_config(config):
    return os.path.join(
            config["sklearn_experiment"]["cache_directory"],
            config["sklearn_experiment"]["model"]["key"],
            config["sklearn_experiment"]["name"],
            "sklearn_objects")


def pipeline_to_disk(config, sklearn_objects):
    joblib_dir = joblib_dir_from_config(config)
    os.makedirs(joblib_dir, exist_ok=True)
    for key, obj in sklearn_objects.items():
        joblib_fname = os.path.join(joblib_dir, key + ".joblib")
        logger.info("Writing scikit-learn object '%s' to '%s'", obj, joblib_fname)
        joblib.dump(obj, joblib_fname)
    return joblib_dir


def pipeline_from_disk(config):
    joblib_dir = joblib_dir_from_config(config)
    if not os.path.isdir(joblib_dir):
        logger.error("Directory '%s' does not exist, cannot load pipeline from disk", joblib_dir)
        return {}
    sklearn_objects = {}
    for f in os.scandir(joblib_dir):
        if not f.name.endswith(".joblib"):
            continue
        logger.info("Loading scikit-learn object from file '%s'", f.path)
        key = f.name.split(".joblib")[0]
        sklearn_objects[key] = joblib.load(f.path)
    return sklearn_objects
