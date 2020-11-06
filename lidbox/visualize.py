import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Modified from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def draw_confusion_matrix(cm, label_names, title='', cmap=plt.cm.Blues, no_legend=True):
    plt.style.use("default")
    num_labels = len(label_names)
    cm = np.array(cm, dtype=np.float32)
    assert cm.shape[1] == cm.shape[0] == num_labels, "Confusion matrix shape {} must match amount of labels {} both in columns and rows".format(cm.shape, num_labels)
    # Normalize by support
    cm /= cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    if title:
        ax.set_title(title)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if not no_legend:
        ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(num_labels),
           yticks=np.arange(num_labels),
           xlim=(-0.5, num_labels - 0.5),
           ylim=(num_labels - 0.5, -0.5),
           xticklabels=label_names,
           yticklabels=label_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for row in range(num_labels):
        for col in range(num_labels):
            ax.text(col, row, format(cm[row, col], '.2f'),
                    ha="center",
                    va="center",
                    color="white" if cm[row, col] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return fig, ax


def plot_duration_distribution(metadata):
    sns.set(rc={'figure.figsize': (8, 6)})

    split_names = sorted(metadata.split.unique())
    label_names = sorted(metadata.label.unique())

    ax = sns.boxplot(
        x="split",
        order=split_names,
        y="duration",
        hue="label",
        hue_order=label_names,
        data=metadata)
    ax.set_title("Median audio file duration in seconds")
    plt.show()

    ax = sns.barplot(
        x="split",
        order=split_names,
        y="duration",
        hue="label",
        hue_order=label_names,
        data=metadata,
        ci=None,
        estimator=np.sum)
    ax.set_title("Total amount of audio in seconds")
    plt.show()


def plot_signal(signal, figsize=(6, 0.5), **kwargs):
    ax = sns.lineplot(data=signal, lw=0.1, **kwargs)
    ax.set_axis_off()
    ax.margins(0)
    plt.gcf().set_size_inches(*figsize)
    plt.show()


def plot_spectrogram(S, cmap="viridis", figsize=None, **kwargs):
    if figsize is None:
        figsize = S.shape[0]/50, S.shape[1]/50
    ax = sns.heatmap(S.T, cbar=False, cmap=cmap, **kwargs)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.margins(0)
    plt.gcf().set_size_inches(*figsize)
    plt.show()


def plot_cepstra(X, figsize=None):
    if not figsize:
        figsize = (X.shape[0]/50, X.shape[1]/20)
    plot_spectrogram(X, cmap="RdBu_r", figsize=figsize)
